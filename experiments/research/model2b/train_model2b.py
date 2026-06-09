#!/usr/bin/env python3
"""
Model 2B "The Bettor" — gap zone calibration model.

For every fight (men's, 2018+):
  - Compute M1 prob (full production M1, no OOF)
  - Compute M2A prob (full production M2A, no OOF)
  - Compute gap = m2a_pick_prob - m2a_pick_novig (pos = M2A more bullish than Vegas)
  - target = did M2A's predicted winner actually win?

Run from project root:
    python experiments/research/model2b/train_model2b.py
"""

import bisect, gc, json, math, os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR  = Path('experiments/research/model2b')
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
SEED         = 42
FEAT_2B      = ['gap_size','gap_zone','gap_direction','m1_prob','m2a_prob',
                'm1_confidence','m2a_confidence','m1_m2a_agree','vegas_agree',
                'triple_agree','odds_tier','weight_class_ord','is_5round','vig','closing_odds']

ZONE_LABELS  = {0:'Lock',1:'Strong',2:'Lean',3:'Watch',4:'Value',5:'Strong Value',6:'Max Value'}

WOMENS_CLASSES = {
    "Women's Strawweight","Women's Flyweight",
    "Women's Bantamweight","Women's Featherweight",
}
WC_ORDER = {
    "Women's Strawweight":0,"Women's Flyweight":1,
    "Women's Bantamweight":2,"Women's Featherweight":3,
    "Flyweight":4,"Bantamweight":5,"Featherweight":6,
    "Lightweight":7,"Welterweight":8,"Middleweight":9,
    "Light Heavyweight":10,"Heavyweight":11,"Catch Weight":6,
}

print("=" * 65)
print("MODEL 2B — The Bettor (gap zone calibration)")
print("=" * 65)

# ── Load models ───────────────────────────────────────────────────────────────
print("\n[SETUP] Loading models...")
model_lr_m1  = joblib.load('model/ufc_model_best.pkl')
model_xgb_m1 = joblib.load('model/ufc_model_xgb.pkl')
feat_cols_m1 = joblib.load('model/feature_columns_best.pkl')
model_lr_m2a  = joblib.load('model/ufc_model2a_best.pkl')
model_xgb_m2a = joblib.load('model/ufc_model2a_xgb.pkl')
feat_cols_m2a = joblib.load('model/ufc_model2a_features.pkl')
with open('model/model2a_tier_stats.json') as f:
    tier_stats = json.load(f)
print(f"  M1: {len(feat_cols_m1)} features | M2A: {len(feat_cols_m2a)} features")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[SETUP] Loading data...")
df_master  = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_master['date'] = pd.to_datetime(df_master['date'])
career_raw = pd.read_csv('data/career_fights_updated.csv')
career_raw['date'] = pd.to_datetime(career_raw['date'])
career_raw = career_raw.sort_values(['fighter','date']).reset_index(drop=True)
fstats_df  = pd.read_csv('data/ufc_fighters_final_updated.csv')
for c in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    fstats_df[c] = pd.to_numeric(
        fstats_df[c].astype(str).str.replace('%','',regex=False),
        errors='coerce').fillna(0) / 100.0
elo_hist = pd.read_csv('data/elo_ratings_history.csv')
elo_hist['date'] = pd.to_datetime(elo_hist['date'])
elo_hist = elo_hist.sort_values(['fighter','date']).reset_index(drop=True)

df = df_master[
    (df_master['date'] >= '2018-01-01') &
    df_master['R_odds'].notna() & df_master['B_odds'].notna() &
    df_master['r_dec_odds'].notna() & df_master['b_dec_odds'].notna() &
    df_master['r_sub_odds'].notna() & df_master['b_sub_odds'].notna() &
    df_master['r_ko_odds'].notna()  & df_master['b_ko_odds'].notna() &
    df_master['Winner'].isin(['Red','Blue']) &
    ~df_master['weight_class'].isin(WOMENS_CLASSES)
].copy().reset_index(drop=True)
print(f"  Universe (men's, 2018+, all odds): {len(df)} fights")

# Corner randomization — seed=42 matches M2A training exactly
np.random.seed(SEED)
swap_mask = np.random.random(len(df)) < 0.5
r_matched = sorted([c for c in df.columns if c.startswith('R_') and ('B_'+c[2:]) in df.columns])
b_matched  = ['B_'+c[2:] for c in r_matched]
for rc, bc in zip(r_matched, b_matched):
    rv = df.loc[swap_mask, rc].values.copy(); bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv; df.loc[swap_mask, bc] = rv
df.loc[swap_mask & (df['Winner']=='Red'),  'Winner'] = 'TEMP'
df.loc[swap_mask & (df['Winner']=='Blue'), 'Winner'] = 'Red'
df.loc[swap_mask & (df['Winner']=='TEMP'), 'Winner'] = 'Blue'
for rc, bc in [('r_dec_odds','b_dec_odds'),('r_sub_odds','b_sub_odds'),('r_ko_odds','b_ko_odds')]:
    rv = df.loc[swap_mask, rc].values.copy(); bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv; df.loc[swap_mask, bc] = rv

target_red  = (df['Winner'] == 'Red').astype(int).values
train_mask  = (df['date'] < TRAIN_CUTOFF).values
test_mask   = ~train_mask
train_idx   = np.where(train_mask)[0]
test_idx    = np.where(test_mask)[0]
print(f"  Train 2018-2023: {len(train_idx)} | Test 2024+: {len(test_idx)}")

# ── Career stats ──────────────────────────────────────────────────────────────
print("\n[SETUP] Building career stats...")
cf = career_raw.copy()
def shift_cumsum(x): return x.cumsum().shift(1).fillna(0)
cf['cum_fights']      = cf.groupby('fighter').cumcount()
cf['cum_wins']        = cf.groupby('fighter')['won'].transform(shift_cumsum)
cf['career_win_rate'] = np.where(cf['cum_fights']>0, cf['cum_wins']/cf['cum_fights'], 0.5)
cf['ko_win']  = ((cf['won']==1)&cf['method'].str.contains('KO|TKO',case=False,na=False)).astype(int)
cf['sub_win'] = ((cf['won']==1)&cf['method'].str.contains('Sub|Submission',case=False,na=False)).astype(int)
cf['fin_win'] = ((cf['won']==1)&cf['method'].str.contains('KO|TKO|Sub|Submission',case=False,na=False)).astype(int)
cf['cum_ko']  = cf.groupby('fighter')['ko_win'].transform(shift_cumsum)
cf['cum_sub'] = cf.groupby('fighter')['sub_win'].transform(shift_cumsum)
cf['ko_finish_rate']  = np.where(cf['cum_fights']>0, cf['cum_ko']/cf['cum_fights'], 0.0)
cf['sub_finish_rate'] = np.where(cf['cum_fights']>0, cf['cum_sub']/cf['cum_fights'], 0.0)
def roll_sh(x, n): return x.shift(1).rolling(n, min_periods=1).mean()
cf['last3_win_rate']    = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x,3)).fillna(0.5)
cf['last5_won']         = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x,5)).fillna(0.5)
cf['last10_win_rate']   = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x,10)).fillna(0.5)
cf['last5_finish_rate'] = cf.groupby('fighter')['fin_win'].transform(lambda x: roll_sh(x,5)).fillna(0.0)
cf['trend_score']       = cf['last3_win_rate'] - cf['last10_win_rate']
cf['prev_date']         = cf.groupby('fighter')['date'].shift(1)
cf['layoff_days']       = (cf['date']-cf['prev_date']).dt.days.fillna(365.0)
wr_cache = cf.groupby('fighter')['won'].mean().to_dict()
def opp_quality_series(grp):
    opps=grp['opponent'].values; res=np.full(len(grp),0.5)
    for i in range(len(grp)):
        prior=opps[max(0,i-5):i]; rates=[wr_cache.get(o,0.5) for o in prior]
        res[i]=float(np.mean(rates)) if rates else 0.5
    return pd.Series(res, index=grp.index)
cf['opp_quality'] = cf.groupby('fighter', group_keys=False).apply(opp_quality_series)

CAREER_COLS = ['cum_fights','career_win_rate','ko_finish_rate','sub_finish_rate',
               'last3_win_rate','last5_won','last10_win_rate','last5_finish_rate',
               'trend_score','layoff_days','opp_quality']
DEFAULT_CAREER = {'cum_fights':0,'career_win_rate':0.5,'ko_finish_rate':0.0,
                  'sub_finish_rate':0.0,'last3_win_rate':0.5,'last5_won':0.5,
                  'last10_win_rate':0.5,'last5_finish_rate':0.0,'trend_score':0.0,
                  'layoff_days':365.0,'opp_quality':0.5}
career_by_f={}; career_dates_f={}
for fname, grp in cf.groupby('fighter'):
    g_ = grp.reset_index(drop=True)
    career_by_f[fname]=g_; career_dates_f[fname]=g_['date'].tolist()
def get_career_at(fighter, fdate):
    if fighter not in career_by_f: return DEFAULT_CAREER.copy()
    idx = bisect.bisect_right(career_dates_f[fighter], fdate) - 1
    if idx < 0: return DEFAULT_CAREER.copy()
    return {c: float(career_by_f[fighter].iloc[idx][c]) for c in CAREER_COLS}
elo_by_f={}; elo_dates_f={}
for fname, grp in elo_hist.groupby('fighter'):
    g_ = grp.sort_values('date').reset_index(drop=True)
    elo_by_f[fname]=g_; elo_dates_f[fname]=g_['date'].tolist()
def get_elo_at(fighter, fdate):
    if fighter not in elo_by_f: return {'elo':1500.0,'elo_trend':0.0}
    idx = bisect.bisect_left(elo_dates_f[fighter], fdate) - 1
    if idx < 0: return {'elo':1500.0,'elo_trend':0.0}
    row = elo_by_f[fighter].iloc[idx]
    return {'elo':float(row['elo_after']),'elo_trend':float(row.get('elo_trend',0.0) or 0.0)}
fstyle={}
for _, row in fstats_df.iterrows():
    fstyle[row['Fighter_Name']] = {k:float(row.get(k,0) or 0)
        for k in ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']}

def g(row, col, default=0.0):
    v = row.get(col,default) if isinstance(row,dict) else getattr(row,col,default)
    try:
        if pd.isna(v): return float(default)
    except Exception: pass
    return float(v) if v is not None else float(default)
def layoff_buckets(days):
    return {'lt90':1 if days<90 else 0,'90_180':1 if 90<=days<180 else 0,
            '180_365':1 if 180<=days<365 else 0,'gt365':1 if days>=365 else 0}

# ── Build 129-feature M1 matrix ───────────────────────────────────────────────
print("\n[SETUP] Building 129-feature M1 matrix...")
def build_m1_row(df_row):
    rn=df_row['R_fighter']; bn=df_row['B_fighter']; fd=df_row['date']
    rc=get_career_at(rn,fd); bc=get_career_at(bn,fd)
    rs=fstyle.get(rn,{}); bs=fstyle.get(bn,{})
    re=get_elo_at(rn,fd); be=get_elo_at(bn,fd)
    rlb=layoff_buckets(rc['layoff_days']); blb=layoff_buckets(bc['layoff_days'])
    rsp=1 if str(df_row.get('R_Stance','') or '').lower()=='southpaw' else 0
    bsp=1 if str(df_row.get('B_Stance','') or '').lower()=='southpaw' else 0
    rw=g(df_row,'R_wins'); bw=g(df_row,'B_wins')
    rl=g(df_row,'R_losses'); bl=g(df_row,'B_losses')
    rh=g(df_row,'R_Height_cms',175); bh=g(df_row,'B_Height_cms',175)
    rrch=g(df_row,'R_Reach_cms',175); brch=g(df_row,'B_Reach_cms',175)
    ra=g(df_row,'R_age',28); ba=g(df_row,'B_age',28)
    rsig=g(df_row,'R_avg_SIG_STR_landed'); bsig=g(df_row,'B_avg_SIG_STR_landed')
    rtd=g(df_row,'R_avg_TD_landed'); btd=g(df_row,'B_avg_TD_landed')
    rws=g(df_row,'R_current_win_streak'); bws=g(df_row,'B_current_win_streak')
    rls=g(df_row,'R_current_lose_streak'); bls=g(df_row,'B_current_lose_streak')
    rlws=g(df_row,'R_longest_win_streak'); blws=g(df_row,'B_longest_win_streak')
    rsp2=g(df_row,'R_avg_SIG_STR_pct'); bsp2=g(df_row,'B_avg_SIG_STR_pct')
    rsba=g(df_row,'R_avg_SUB_ATT'); bsba=g(df_row,'B_avg_SUB_ATT')
    rtdp=g(df_row,'R_avg_TD_pct'); btdp=g(df_row,'B_avg_TD_pct')
    bttb=g(df_row,'B_total_title_bouts'); rttb=g(df_row,'R_total_title_bouts')
    rko=g(df_row,'R_win_by_KO/TKO'); bko=g(df_row,'B_win_by_KO/TKO')
    rsub=g(df_row,'R_win_by_Submission'); bsub=g(df_row,'B_win_by_Submission')
    wc_ord=WC_ORDER.get(str(df_row.get('weight_class','') or ''),6)
    raxe=ra*rc['cum_fights']; baxe=ba*bc['cum_fights']
    rqawr=rc['career_win_rate']; bqawr=bc['career_win_rate']
    rqafr=rc['last5_finish_rate']; bqafr=bc['last5_finish_rate']
    rlc=min(rc['layoff_days'],730); blc=min(bc['layoff_days'],730)
    rfd=rc['ko_finish_rate']+rc['sub_finish_rate']; bfd=bc['ko_finish_rate']+bc['sub_finish_rate']
    ttr=rw+rl; ttb2=bw+bl
    rgfr=(rl/ttr)*0.5 if ttr>0 else 0.5; bgfr=(bl/ttb2)*0.5 if ttb2>0 else 0.5
    return {
        'R_wins':rw,'R_losses':rl,'R_Height_cms':rh,'R_age':ra,
        'R_avg_SIG_STR_landed':rsig,'R_avg_TD_landed':rtd,
        'R_current_win_streak':rws,'R_current_lose_streak':rls,
        'R_longest_win_streak':rlws,'R_avg_SIG_STR_pct':rsp2,
        'R_avg_SUB_ATT':rsba,'R_avg_TD_pct':rtdp,'R_Reach_cms':rrch,
        'B_wins':bw,'B_losses':bl,'B_Height_cms':bh,'B_age':ba,
        'B_avg_SIG_STR_landed':bsig,'B_avg_TD_landed':btd,
        'B_current_win_streak':bws,'B_current_lose_streak':bls,
        'B_longest_win_streak':blws,'B_avg_SIG_STR_pct':bsp2,
        'B_avg_SUB_ATT':bsba,'B_avg_TD_pct':btdp,'B_Reach_cms':brch,
        'B_total_title_bouts':bttb,
        'win_dif':rw-bw,'loss_dif':rl-bl,'win_streak_dif':rws-bws,'lose_streak_dif':rls-bls,
        'height_dif':rh-bh,'reach_dif':rrch-brch,'age_dif':ra-ba,
        'sig_str_dif':rsig-bsig,'avg_td_dif':rtd-btd,'ko_dif':rko-bko,'sub_dif':rsub-bsub,
        'total_title_bout_dif':rttb-bttb,'weight_class_ord':wc_ord,
        'orth_clash':1 if(rsp==0 and bsp==0) else 0,'south_clash':1 if(rsp==1 and bsp==1) else 0,
        'R_southpaw':rsp,'R_cum_fights':rc['cum_fights'],'B_cum_fights':bc['cum_fights'],
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
        'R_age_x_exp':raxe,'B_age_x_exp':baxe,'age_x_exp_dif':raxe-baxe,
        'R_layoff_lt90':rlb['lt90'],'R_layoff_90_180':rlb['90_180'],
        'R_layoff_180_365':rlb['180_365'],'R_layoff_gt365':rlb['gt365'],
        'B_layoff_lt90':blb['lt90'],'B_layoff_90_180':blb['90_180'],'B_layoff_180_365':blb['180_365'],
        'R_SLpM':rs.get('SLpM',0),'R_SApM':rs.get('SApM',0),
        'R_Str_Acc':rs.get('Str_Acc',0),'R_Str_Def':rs.get('Str_Def',0),
        'R_TD_Avg':rs.get('TD_Avg',0),'R_TD_Acc':rs.get('TD_Acc',0),
        'R_TD_Def':rs.get('TD_Def',0),'R_Sub_Avg':rs.get('Sub_Avg',0),
        'B_SLpM':bs.get('SLpM',0),'B_SApM':bs.get('SApM',0),
        'B_Str_Acc':bs.get('Str_Acc',0),'B_Str_Def':bs.get('Str_Def',0),
        'B_TD_Avg':bs.get('TD_Avg',0),'B_TD_Acc':bs.get('TD_Acc',0),
        'B_TD_Def':bs.get('TD_Def',0),'B_Sub_Avg':bs.get('Sub_Avg',0),
        'SLpM_dif':rs.get('SLpM',0)-bs.get('SLpM',0),'SApM_dif':rs.get('SApM',0)-bs.get('SApM',0),
        'Str_Def_dif':rs.get('Str_Def',0)-bs.get('Str_Def',0),
        'TD_Def_dif':rs.get('TD_Def',0)-bs.get('TD_Def',0),
        'Sub_Avg_dif':rs.get('Sub_Avg',0)-bs.get('Sub_Avg',0),
        'TD_Avg_dif':rs.get('TD_Avg',0)-bs.get('TD_Avg',0),
        'R_elo':re['elo'],'B_elo':be['elo'],'elo_dif':re['elo']-be['elo'],
        'R_elo_trend':re['elo_trend'],'B_elo_trend':be['elo_trend'],
        'elo_trend_dif':re['elo_trend']-be['elo_trend'],
        'R_qa_win_rate':rqawr,'R_qa_finish_rate':rqafr,'R_qa_SLpM':0.0,'R_qa_SApM':0.0,
        'B_qa_win_rate':bqawr,'B_qa_finish_rate':bqafr,'B_qa_SLpM':0.0,'B_qa_SApM':0.0,
        'qa_win_rate_dif':rqawr-bqawr,'qa_finish_rate_dif':rqafr-bqafr,'qa_SLpM_dif':0.0,'qa_SApM_dif':0.0,
        'R_age_x_layoff':ra*rlc,'B_age_x_layoff':ba*blc,'age_x_layoff_dif':ra*rlc-ba*blc,
        'R_finish_danger':rfd,'B_finish_danger':bfd,
        'finish_danger_mismatch':rfd*bgfr-bfd*rgfr,
        'R_got_finished_rate':rgfr,'B_got_finished_rate':bgfr,
    }

rows_list = [build_m1_row(row) for _, row in df.iterrows()]
X_m1 = pd.DataFrame(rows_list, columns=feat_cols_m1).values.astype(float)
cm = np.nanmedian(X_m1,axis=0); nm=np.isnan(X_m1); X_m1[nm]=np.take(cm,np.where(nm)[1])
print(f"  M1 matrix: {X_m1.shape}")
gc.collect()

# ── Generate M1 probs (full production, no OOF) ───────────────────────────────
print("\n[M1] Generating probabilities...")
m1_probs_all = (0.70 * model_lr_m1.predict_proba(X_m1)[:,1] +
                0.30 * model_xgb_m1.predict_proba(X_m1)[:,1])
gc.collect()

# ── Build 42-feature M2A matrix ───────────────────────────────────────────────
print("\n[M2A] Building 42-feature matrix to generate M2A probs...")

def implied_prob(odds):
    try:
        odds = float(odds)
        if odds == 0 or np.isnan(odds): return None
        return abs(odds)/(abs(odds)+100) if odds < 0 else 100/(odds+100)
    except Exception: return None

def novig_probs(f1_odds, f2_odds):
    f1_raw=implied_prob(f1_odds) or 0.5; f2_raw=implied_prob(f2_odds) or 0.5
    total=f1_raw+f2_raw
    if total<=0: return 0.5,0.5,0.0
    return f1_raw/total, f2_raw/total, total-1.0

BASE_M2 = ['model1_prob','f1_ml_novig','f2_ml_novig','ml_gap','vig',
           'f1_dec_implied','f2_dec_implied','dec_implied_dif',
           'f1_ko_implied','f2_ko_implied','ko_implied_dif',
           'f1_sub_implied','f2_sub_implied','sub_implied_dif',
           'finish_prob','f1_finish_prob','f2_finish_prob','finish_advantage',
           'abs_gap','vegas_confidence','model_confidence','model_agrees_vegas','gap_x_confidence']

novig_f1_all = np.zeros(len(df))
novig_f2_all = np.zeros(len(df))
vig_all      = np.zeros(len(df))
r_odds_all   = df['R_odds'].values.astype(float)
b_odds_all   = df['B_odds'].values.astype(float)

m2_rows = []
for i, (_, dr) in enumerate(df.iterrows()):
    m1p = float(m1_probs_all[i])
    f1n, f2n, vig_ = novig_probs(dr['R_odds'], dr['B_odds'])
    novig_f1_all[i] = f1n; novig_f2_all[i] = f2n; vig_all[i] = vig_
    ml_gap_ = m1p - f1n
    f1_dec=implied_prob(dr['r_dec_odds']) or 0.0; f2_dec=implied_prob(dr['b_dec_odds']) or 0.0
    f1_ko=implied_prob(dr['r_ko_odds'])   or 0.0; f2_ko=implied_prob(dr['b_ko_odds'])   or 0.0
    f1_sub=implied_prob(dr['r_sub_odds']) or 0.0; f2_sub=implied_prob(dr['b_sub_odds']) or 0.0
    dec_tot=f1_dec+f2_dec; fin_p=1.0-(dec_tot/2.0) if dec_tot>0 else 0.5
    f1_fin=f1_ko+f1_sub; f2_fin=f2_ko+f2_sub
    m2_rows.append([m1p,f1n,f2n,ml_gap_,vig_,f1_dec,f2_dec,f1_dec-f2_dec,
                    f1_ko,f2_ko,f1_ko-f2_ko,f1_sub,f2_sub,f1_sub-f2_sub,
                    fin_p,f1_fin,f2_fin,f1_fin-f2_fin,abs(ml_gap_),abs(f1n-0.5),abs(m1p-0.5),
                    1 if (m1p>0.5)==(f1n>0.5) else 0, ml_gap_*abs(f1n-0.5)])

X2_base = np.array(m2_rows,dtype=float)
cm2=np.nanmedian(X2_base,axis=0); nm2=np.isnan(X2_base); X2_base[nm2]=np.take(cm2,np.where(nm2)[1])

# Step 1: fav/dog profile + tier (tier from training rows only — same as M2A training)
df_fh = df[['date','R_fighter','B_fighter','R_odds','B_odds','Winner']].copy()
df_fh['f1_won'] = (df_fh['Winner']=='Red').astype(int)
df_sorted = df_fh.sort_values('date').reset_index()
fav_bouts={}; fav_wins={}; dog_bouts={}; dog_wins={}
tier_train_counts={t:0 for t in range(5)}; tier_train_wins={t:0 for t in range(5)}
step1_rows=[None]*len(df)
train_set = set(train_idx.tolist())
odds_tier_all = np.zeros(len(df), dtype=int)
for _, row in df_sorted.iterrows():
    orig_i=row['index']; f1=row['R_fighter']; f1_won=row['f1_won']
    f1nv,_,_ = novig_probs(df.loc[orig_i,'R_odds'], df.loc[orig_i,'B_odds'])
    is_fav = 1 if df.loc[orig_i,'R_odds'] < 0 else 0
    t = 0 if f1nv<0.30 else (1 if f1nv<0.45 else (2 if f1nv<0.55 else (3 if f1nv<0.70 else 4)))
    odds_tier_all[orig_i] = t
    fav_n=len(fav_bouts.get(f1,[])); fav_w=fav_wins.get(f1,0)
    dog_n=len(dog_bouts.get(f1,[])); dog_w=dog_wins.get(f1,0)
    step1_rows[orig_i]=[is_fav,fav_w/fav_n if fav_n>0 else 0.5,dog_w/dog_n if dog_n>0 else 0.5,
                        math.log1p(fav_n),math.log1p(dog_n),abs(f1nv-0.5),t]
    if orig_i in train_set:
        tier_train_counts[t]+=1; tier_train_wins[t]+=f1_won
    if is_fav:
        fav_bouts.setdefault(f1,[]).append(row['date']); fav_wins[f1]=fav_wins.get(f1,0)+f1_won
    else:
        dog_bouts.setdefault(f1,[]).append(row['date']); dog_wins[f1]=dog_wins.get(f1,0)+f1_won

tier_wr_map = {t: tier_train_wins[t]/tier_train_counts[t]
               if tier_train_counts[t]>0 else 0.5 for t in range(5)}
step1_arr = np.array(step1_rows,dtype=float)
tier_hist_wr = np.array([float(tier_wr_map.get(int(t),0.5)) for t in step1_arr[:,6]])
step1_final  = np.column_stack([step1_arr[:,:6], tier_hist_wr])

# Step 2: method × style interactions
feat_idx = {f:i for i,f in enumerate(feat_cols_m1)}
step2_rows=[]
for i in range(len(df)):
    br=X2_base[i]
    f1ko=br[BASE_M2.index('f1_ko_implied')]; f2ko=br[BASE_M2.index('f2_ko_implied')]
    f1sub=br[BASE_M2.index('f1_sub_implied')]; f2sub=br[BASE_M2.index('f2_sub_implied')]
    f1dec=br[BASE_M2.index('f1_dec_implied')]; f2dec=br[BASE_M2.index('f2_dec_implied')]
    m1p_=br[BASE_M2.index('model1_prob')]; fin_p_=br[BASE_M2.index('finish_prob')]
    rkofr=X_m1[i,feat_idx['R_ko_finish_rate']]; bkofr=X_m1[i,feat_idx['B_ko_finish_rate']]
    rsubfr=X_m1[i,feat_idx['R_sub_finish_rate']]; bsubfr=X_m1[i,feat_idx['B_sub_finish_rate']]
    strdef=X_m1[i,feat_idx['Str_Def_dif']]
    step2_rows.append([f1ko*rkofr-f2ko*bkofr, f1sub*rsubfr-f2sub*bsubfr,
                       fin_p_*abs(m1p_-0.5), ((f1dec+f2dec)/2.0)*abs(strdef),
                       f1ko+f2ko, f1sub+f2sub, abs(f1ko-f2ko), abs(f1sub-f2sub)])
step2_arr=np.array(step2_rows,dtype=float)

# Step 3: weight-class context
wc_arr=X_m1[:,feat_idx['weight_class_ord']]
no_rds=df.get('no_of_rounds', pd.Series([3]*len(df))).fillna(3).values
m1_wc_acc={}
for wc_v in np.unique(wc_arr[train_mask]):
    mask = train_mask & (wc_arr==wc_v)
    if mask.sum()>=5:
        m1_wc_acc[int(wc_v)]=accuracy_score(target_red[mask],(m1_probs_all[mask]>0.5).astype(int))
m1_train_acc_global = accuracy_score(target_red[train_idx],(m1_probs_all[train_idx]>0.5).astype(int))
step3_rows=[]
for i in range(len(df)):
    wc_v=wc_arr[i]; is_5r=1 if no_rds[i]>=5 else 0
    wca=m1_wc_acc.get(int(wc_v),m1_train_acc_global)
    m1c=abs(X2_base[i,BASE_M2.index('model_confidence')])
    step3_rows.append([wc_v/11.0, is_5r, wca-m1_train_acc_global, is_5r*m1c])
step3_arr=np.array(step3_rows,dtype=float)

X2_full = np.hstack([X2_base, step1_final, step2_arr, step3_arr])
assert X2_full.shape[1] == 42

# ── Generate M2A probs (full production M2A) ──────────────────────────────────
print("[M2A] Generating probabilities with full production M2A...")
m2a_lr_p  = model_lr_m2a.predict_proba(X2_full)[:,1]
m2a_xgb_p = model_xgb_m2a.predict_proba(X2_full)[:,1]
m2a_probs_all = 0.50 * m2a_lr_p + 0.50 * m2a_xgb_p
gc.collect()

# ── Compute gap features for all fights ───────────────────────────────────────
print("\n[M2B] Computing gap features...")

def gap_zone_fn(gap_size):
    if gap_size < 0.01:  return 0
    elif gap_size < 0.02: return 1
    elif gap_size < 0.03: return 2
    elif gap_size < 0.05: return 3
    elif gap_size < 0.08: return 4
    elif gap_size < 0.10: return 5
    else:                 return 6

m2a_picks_red = (m2a_probs_all > 0.5).astype(int)
m1_picks_red  = (m1_probs_all  > 0.5).astype(int)
vegas_fav_red = (novig_f1_all  > 0.5).astype(int)

# pick_prob = M2A's confidence in its predicted winner
pick_prob_all  = np.where(m2a_picks_red==1, m2a_probs_all, 1.0-m2a_probs_all)
pick_novig_all = np.where(m2a_picks_red==1, novig_f1_all,  novig_f2_all)

gap_all      = pick_prob_all - pick_novig_all
gap_size_all = np.abs(gap_all)
# Trivariate direction: 1=pos_gap (model more confident than Vegas, ROI +35.8% agree),
# -1=neg_gap (Vegas more confident, ROI -4.9% agree), 0=near_zero (|gap|<1%)
# Largest directional split in dataset: pos_gap agree 83.2% WR vs neg_gap agree 71.4%
gap_dir_all  = np.where(gap_all > 0.01, 1, np.where(gap_all < -0.01, -1, 0))
gap_zone_all = np.array([gap_zone_fn(g) for g in gap_size_all])

# target: did M2A's predicted winner win?
# target_red = 1 if Red won; m2a_picks_red = 1 if M2A predicts Red
target_m2b = (target_red == m2a_picks_red).astype(int)

# closing_odds for M2A's pick (American odds)
closing_odds_all = np.where(m2a_picks_red==1, r_odds_all, b_odds_all)

m1_m2a_agree_all = (m1_picks_red == m2a_picks_red).astype(int)
vegas_agree_all  = (m2a_picks_red == vegas_fav_red).astype(int)
triple_agree_all = (m1_m2a_agree_all & vegas_agree_all).astype(int)

# odds_tier from M2A's PICK perspective
# (re-bin based on pick_novig so it reflects favoritism from M2A's angle)
pick_tier_all = np.array([
    0 if pn<0.30 else (1 if pn<0.45 else (2 if pn<0.55 else (3 if pn<0.70 else 4)))
    for pn in pick_novig_all
])

wc_norm_all = wc_arr / 11.0
is_5r_all   = (no_rds >= 5).astype(int)

# Build M2B feature matrix
X2b = np.column_stack([
    gap_size_all,       # gap_size
    gap_zone_all,       # gap_zone
    gap_dir_all,        # gap_direction
    m1_probs_all,       # m1_prob (prob Red wins)
    m2a_probs_all,      # m2a_prob (prob Red wins)
    np.abs(m1_probs_all  - 0.5),  # m1_confidence
    np.abs(m2a_probs_all - 0.5),  # m2a_confidence
    m1_m2a_agree_all,   # m1_m2a_agree
    vegas_agree_all,    # vegas_agree
    triple_agree_all,   # triple_agree
    pick_tier_all,      # odds_tier (from pick's perspective)
    wc_norm_all,        # weight_class_ord
    is_5r_all,          # is_5round
    vig_all,            # vig
    closing_odds_all,   # closing_odds (American, for M2A's pick)
])
assert X2b.shape[1] == len(FEAT_2B)

cm_b=np.nanmedian(X2b,axis=0); nm_b=np.isnan(X2b); X2b[nm_b]=np.take(cm_b,np.where(nm_b)[1])

X2b_train = X2b[train_idx]; X2b_test = X2b[test_idx]
y2b_train  = target_m2b[train_idx]; y2b_test = target_m2b[test_idx]
print(f"  M2B feature matrix: {X2b.shape}")
print(f"  Overall M2A pick win rate: {target_m2b.mean():.4f}")
print(f"  Train pick win rate: {y2b_train.mean():.4f} | Test: {y2b_test.mean():.4f}")
gc.collect()

# ── GAP ZONE ANALYSIS (full dataset) ─────────────────────────────────────────
print("\n" + "=" * 65)
print("GAP ZONE ANALYSIS — full dataset (train + test)")
print("=" * 65)

def unit_return(odds):
    o = float(odds)
    if np.isnan(o) or o == 0: return 0.0
    return 100.0/abs(o) if o < 0 else o/100.0

header = f"{'Zone':>12} {'Label':>12} {'N':>6} {'WR%':>6} {'Agree WR%':>9} {'Triple WR%':>10} {'Disagree WR%':>12} {'AvgOdds':>8} {'ROI%':>7}"
print(header)
print("-"*90)
zone_stats = {}
for z in range(7):
    mask_z = (gap_zone_all == z)
    if mask_z.sum() == 0:
        continue
    wins   = target_m2b[mask_z]
    agree  = m1_m2a_agree_all[mask_z]
    triple = triple_agree_all[mask_z]
    odds_z = closing_odds_all[mask_z]

    n      = mask_z.sum()
    wr     = wins.mean()
    agree_mask = agree==1; disagree_mask = agree==0
    wr_ag  = wins[agree_mask].mean() if agree_mask.sum()>0 else float('nan')
    wr_tri = wins[triple==1].mean()  if (triple==1).sum()>0 else float('nan')
    wr_dis = wins[disagree_mask].mean() if disagree_mask.sum()>0 else float('nan')
    avg_odds = np.nanmedian(odds_z)
    roi = np.mean([unit_return(o)*(w*2-1) for o, w in zip(odds_z, wins)])*100
    print(f"  {z:>6} {ZONE_LABELS[z]:>12} {n:>6} {wr:>6.3f} {wr_ag:>9.3f} {wr_tri:>10.3f} {wr_dis:>12.3f} {avg_odds:>8.0f} {roi:>7.1f}%")
    zone_stats[z] = {'n':int(n),'win_rate':float(wr),
                     'agree_wr':float(wr_ag) if not np.isnan(wr_ag) else None,
                     'triple_wr':float(wr_tri) if not np.isnan(wr_tri) else None,
                     'disagree_wr':float(wr_dis) if not np.isnan(wr_dis) else None,
                     'avg_closing_odds':float(avg_odds),'roi_pct':float(roi)}
print("-"*90)

# Overall agreement analysis
for label, mask_ in [
    ("All",            np.ones(len(df),dtype=bool)),
    ("M1+M2A agree",   m1_m2a_agree_all==1),
    ("Triple agree",   triple_agree_all==1),
    ("M1+M2A disagree",m1_m2a_agree_all==0),
]:
    n_ = mask_.sum()
    if n_ == 0: continue
    wr_ = target_m2b[mask_].mean()
    odds_ = closing_odds_all[mask_]
    roi_ = np.mean([unit_return(o)*(w*2-1) for o, w in zip(odds_, target_m2b[mask_])])*100
    print(f"  {label:>20}: N={n_:5d}  WR={wr_:.3f}  ROI={roi_:+.1f}%")

# ── TRAIN M2B ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TRAINING M2B")
print("=" * 65)

# LR with isotonic calibration
print("\n[M2B] LR (calibrated isotonic)...")
base_lr = Pipeline([('sc', RobustScaler()),
                    ('clf', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                               max_iter=2000, random_state=SEED))])
m2b_lr = CalibratedClassifierCV(base_lr, cv=5, method='isotonic')
m2b_lr.fit(X2b_train, y2b_train)
lr_cv = cross_val_score(base_lr, X2b_train, y2b_train,
                        cv=StratifiedKFold(5,shuffle=True,random_state=SEED), n_jobs=1).mean()
lr_test_acc = accuracy_score(y2b_test, (m2b_lr.predict_proba(X2b_test)[:,1]>0.5).astype(int))
print(f"  LR base CV acc: {lr_cv:.4f} | Calibrated test acc: {lr_test_acc:.4f}")
gc.collect()

# Random Forest
print("\n[M2B] Random Forest (200 trees, depth 5)...")
m2b_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED, n_jobs=1)
m2b_rf.fit(X2b_train, y2b_train)
rf_test_acc = accuracy_score(y2b_test, m2b_rf.predict(X2b_test))
rf_brier    = brier_score_loss(y2b_test, m2b_rf.predict_proba(X2b_test)[:,1])
print(f"  RF test acc: {rf_test_acc:.4f} | Brier: {rf_brier:.4f}")
gc.collect()

# XGBoost with Optuna
print("\n[M2B] XGBoost (15 Optuna trials)...")
def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators',100,400),
        'max_depth':        trial.suggest_int('max_depth',2,5),
        'learning_rate':    trial.suggest_float('learning_rate',0.01,0.2,log=True),
        'subsample':        trial.suggest_float('subsample',0.6,1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.6,1.0),
        'min_child_weight': trial.suggest_int('min_child_weight',1,5),
        'gamma':            trial.suggest_float('gamma',0.0,0.5),
        'reg_lambda':       trial.suggest_float('reg_lambda',0.5,3.0),
    }
    clf = XGBClassifier(**params, eval_metric='logloss', random_state=SEED, verbosity=0, n_jobs=1)
    return cross_val_score(clf, X2b_train, y2b_train,
                           cv=StratifiedKFold(5,shuffle=True,random_state=SEED), n_jobs=1).mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(xgb_objective, n_trials=15, show_progress_bar=False)
print(f"  XGB best CV acc: {study.best_value:.4f}")
m2b_xgb = XGBClassifier(**study.best_params, eval_metric='logloss',
                         random_state=SEED, verbosity=0, n_jobs=1)
m2b_xgb.fit(X2b_train, y2b_train)
xgb_test_acc = accuracy_score(y2b_test, m2b_xgb.predict(X2b_test))
xgb_brier    = brier_score_loss(y2b_test, m2b_xgb.predict_proba(X2b_test)[:,1])
print(f"  XGB test acc: {xgb_test_acc:.4f} | Brier: {xgb_brier:.4f}")
gc.collect()

# ── Ensemble M2B (LR + RF + XGB, 1/3 each) ───────────────────────────────────
print("\n[M2B] Ensemble predictions...")
p_lr  = m2b_lr.predict_proba(X2b_test)[:,1]
p_rf  = m2b_rf.predict_proba(X2b_test)[:,1]
p_xgb = m2b_xgb.predict_proba(X2b_test)[:,1]
p_ens = (p_lr + p_rf + p_xgb) / 3.0
ens_acc   = accuracy_score(y2b_test, (p_ens>0.5).astype(int))
ens_brier = brier_score_loss(y2b_test, p_ens)
print(f"  Ensemble test acc: {ens_acc:.4f} | Brier: {ens_brier:.4f}")

# Calibration per gap zone (test set)
print("\n  Calibration by gap zone (test set):")
print(f"  {'Zone':>6} {'Label':>12} {'N':>5} {'M2B Prob':>9} {'Actual WR':>9} {'CalErr':>7}")
for z in range(7):
    mask_z = test_mask & (gap_zone_all==z)
    if mask_z.sum() < 3: continue
    test_z_idx = np.where(mask_z)[0]
    # map to test-set indices
    test_local = np.where(np.isin(test_idx, test_z_idx))[0]
    if len(test_local)==0: continue
    p_z = p_ens[test_local]
    y_z = y2b_test[test_local]
    cal_err = abs(p_z.mean() - y_z.mean())
    print(f"  {z:>6} {ZONE_LABELS[z]:>12} {len(test_local):>5} {p_z.mean():>9.3f} {y_z.mean():>9.3f} {cal_err:>7.3f}")

# Feature importance (XGB)
print("\n  Top-10 feature importances (XGB):")
fi = m2b_xgb.feature_importances_
top10 = np.argsort(fi)[::-1][:10]
for rank, idx in enumerate(top10):
    print(f"  {rank+1:>3}. {FEAT_2B[idx]:<20} {fi[idx]:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
print("\n[M2B] Saving models...")
m2b_bundle = {
    'lr':  m2b_lr,
    'rf':  m2b_rf,
    'xgb': m2b_xgb,
}
joblib.dump(m2b_bundle,   'model/ufc_model2b.pkl')
joblib.dump(FEAT_2B,      'model/ufc_model2b_features.pkl')
print("  Saved: model/ufc_model2b.pkl, model/ufc_model2b_features.pkl")

# ── Build value_bet_log.csv (all 3007 fights) ─────────────────────────────────
print("\n[M2B] Building value_bet_log.csv...")

# Get M2B probs for ALL fights
p_lr_all  = m2b_lr.predict_proba(X2b)[:,1]
p_rf_all  = m2b_rf.predict_proba(X2b)[:,1]
p_xgb_all = m2b_xgb.predict_proba(X2b)[:,1]
p_ens_all = (p_lr_all + p_rf_all + p_xgb_all) / 3.0

def conf_label(p):
    if p >= 0.75:   return 'LOCK'
    elif p >= 0.65: return 'HIGH'
    elif p >= 0.55: return 'MEDIUM'
    else:           return 'LOW'

log_rows = []
for i in range(len(df)):
    row = df.iloc[i]
    m2a_picks_r = bool(m2a_probs_all[i] > 0.5)
    pick_name = row['R_fighter'] if m2a_picks_r else row['B_fighter']
    actual_winner = row['R_fighter'] if row['Winner']=='Red' else row['B_fighter']
    pick_won = int(pick_name == actual_winner)
    log_rows.append({
        'date':               row['date'].strftime('%Y-%m-%d'),
        'f1_name':            row['R_fighter'],
        'f2_name':            row['B_fighter'],
        'weight_class':       row.get('weight_class',''),
        'no_of_rounds':       int(no_rds[i]),
        'm1_prob':            round(float(m1_probs_all[i]),4),
        'm2a_prob':           round(float(m2a_probs_all[i]),4),
        'm2a_pick':           pick_name,
        'pick_novig':         round(float(pick_novig_all[i]),4),
        'gap':                round(float(gap_all[i]),4),
        'gap_size':           round(float(gap_size_all[i]),4),
        'gap_zone':           int(gap_zone_all[i]),
        'gap_zone_label':     ZONE_LABELS[gap_zone_all[i]],
        'gap_direction':      int(gap_dir_all[i]),
        'closing_odds':       float(closing_odds_all[i]),
        'm1_m2a_agree':       int(m1_m2a_agree_all[i]),
        'vegas_agree':        int(vegas_agree_all[i]),
        'triple_agree':       int(triple_agree_all[i]),
        'm2b_win_prob':       round(float(p_ens_all[i]),4),
        'm2b_confidence':     conf_label(p_ens_all[i]),
        'pick_won':           pick_won,
        'split':              'train' if train_mask[i] else 'test',
    })

log_df = pd.DataFrame(log_rows)
log_df.to_csv('data/value_bet_log.csv', index=False)
print(f"  Saved: data/value_bet_log.csv ({len(log_df)} rows)")

# ── Update model_metadata.json ─────────────────────────────────────────────────
print("\n[M2B] Updating model_metadata.json...")
with open('model/model_metadata.json') as f:
    meta = json.load(f)

meta['model2b'] = {
    'model_type':     'M2B_ensemble_LR_RF_XGB_equal',
    'target':         'did_predicted_winner_win (M2A pick)',
    'n_features':     len(FEAT_2B),
    'feature_list':   FEAT_2B,
    'blend_ratio':    '1/3 LR (isotonic) + 1/3 RF + 1/3 XGB',
    'training_universe': '2018+ men\'s fights with all ML + method odds',
    'n_train':        int(len(train_idx)),
    'n_test':         int(len(test_idx)),
    'test_acc_lr':    round(float(lr_test_acc),4),
    'test_acc_rf':    round(float(rf_test_acc),4),
    'test_acc_xgb':   round(float(xgb_test_acc),4),
    'test_acc_ensemble': round(float(ens_acc),4),
    'brier_ensemble': round(float(ens_brier),4),
    'gap_zones':      zone_stats,
    'confidence_labels': {'LOCK':'>75% win prob','HIGH':'65-75%','MEDIUM':'55-65%','LOW':'<55%'},
    'date_trained':   '2026-05-12',
    'xgb_best_params': study.best_params,
}

with open('model/model_metadata.json','w') as f:
    json.dump(meta, f, indent=2)
print("  Updated: model/model_metadata.json")

print("\n" + "=" * 65)
print("TASK 2 DONE — Model 2B saved")
print(f"  Ensemble test acc: {ens_acc:.4f} | Brier: {ens_brier:.4f}")
print(f"  value_bet_log.csv: {len(log_df)} rows")
print("=" * 65)
