#!/usr/bin/env python3
"""
Model 2A retrain — fixes OOF distributional shift.

Root cause of last attempt's failure: OOF M1 train probs had 64.66% accuracy
while production M1 test probs had 71.27% → M2 learned from weak signal,
got confused at test time. Fix: use full production M1 for ALL rows so the
m1_prob distribution is identical at train and test time.

Run from project root:
    python experiments/research/model2_v2/retrain_m2a.py
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR   = Path('experiments/research/model2_v2')
TRAIN_CUTOFF  = pd.Timestamp('2024-01-01')
SEED          = 42
OLD_M2_ACC    = 0.7320
OLD_AGREE_WR  = 0.7430   # win rate at 10% threshold with agreement filter (from last retrain)
AGREE_GATE    = OLD_AGREE_WR + 0.05   # >79.3%

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
print("MODEL 2A RETRAIN — full M1 signals (no OOF distributional shift)")
print("=" * 65)

# ── Load models ───────────────────────────────────────────────────────────────
print("\n[SETUP] Loading models...")
model_lr_m1  = joblib.load('model/ufc_model_best.pkl')
model_xgb_m1 = joblib.load('model/ufc_model_xgb.pkl')
feat_cols_m1 = joblib.load('model/feature_columns_best.pkl')
print(f"  M1: {len(feat_cols_m1)} features loaded")

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
elo_hist = pd.read_csv('data/elo_ratings_history.csv')
elo_hist['date'] = pd.to_datetime(elo_hist['date'])
elo_hist = elo_hist.sort_values(['fighter', 'date']).reset_index(drop=True)

# Filter: men's only, 2018+, all method odds, valid winner
df = df_master[
    (df_master['date'] >= '2018-01-01') &
    df_master['R_odds'].notna() & df_master['B_odds'].notna() &
    df_master['r_dec_odds'].notna() & df_master['b_dec_odds'].notna() &
    df_master['r_sub_odds'].notna() & df_master['b_sub_odds'].notna() &
    df_master['r_ko_odds'].notna()  & df_master['b_ko_odds'].notna() &
    df_master['Winner'].isin(['Red', 'Blue']) &
    ~df_master['weight_class'].isin(WOMENS_CLASSES)
].copy().reset_index(drop=True)
print(f"  Universe (men's, 2018+, all odds): {len(df)} fights")

# Corner randomization — seed=42 matches original sprint
np.random.seed(SEED)
swap_mask = np.random.random(len(df)) < 0.5
r_matched = sorted([c for c in df.columns if c.startswith('R_') and ('B_'+c[2:]) in df.columns])
b_matched  = ['B_'+c[2:] for c in r_matched]
for rc, bc in zip(r_matched, b_matched):
    rv = df.loc[swap_mask, rc].values.copy(); bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv; df.loc[swap_mask, bc] = rv
df.loc[swap_mask & (df['Winner'] == 'Red'),  'Winner'] = 'TEMP'
df.loc[swap_mask & (df['Winner'] == 'Blue'), 'Winner'] = 'Red'
df.loc[swap_mask & (df['Winner'] == 'TEMP'), 'Winner'] = 'Blue'
for rc, bc in [('r_dec_odds','b_dec_odds'), ('r_sub_odds','b_sub_odds'), ('r_ko_odds','b_ko_odds')]:
    rv = df.loc[swap_mask, rc].values.copy(); bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv; df.loc[swap_mask, bc] = rv

target     = (df['Winner'] == 'Red').astype(int).values
train_mask = (df['date'] < TRAIN_CUTOFF).values
test_mask  = ~train_mask
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]
y_train    = target[train_idx]; y_test = target[test_idx]
print(f"  Train 2018-2023: {len(train_idx)} | Test 2024+: {len(test_idx)}")

# ── Career stats ──────────────────────────────────────────────────────────────
print("\n[SETUP] Building career stats...")
cf = career_raw.copy()
def shift_cumsum(x): return x.cumsum().shift(1).fillna(0)
cf['cum_fights']      = cf.groupby('fighter').cumcount()
cf['cum_wins']        = cf.groupby('fighter')['won'].transform(shift_cumsum)
cf['career_win_rate'] = np.where(cf['cum_fights']>0, cf['cum_wins']/cf['cum_fights'], 0.5)
cf['ko_win']  = ((cf['won']==1) & cf['method'].str.contains('KO|TKO', case=False, na=False)).astype(int)
cf['sub_win'] = ((cf['won']==1) & cf['method'].str.contains('Sub|Submission', case=False, na=False)).astype(int)
cf['fin_win'] = ((cf['won']==1) & cf['method'].str.contains('KO|TKO|Sub|Submission', case=False, na=False)).astype(int)
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
cf['layoff_days']       = (cf['date'] - cf['prev_date']).dt.days.fillna(365.0)
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
    career_by_f[fname] = g_; career_dates_f[fname] = g_['date'].tolist()
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
fstyle = {}
for _, row in fstats_df.iterrows():
    fstyle[row['Fighter_Name']] = {k:float(row.get(k,0) or 0)
        for k in ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']}

def g(row, col, default=0.0):
    v = row.get(col, default) if isinstance(row, dict) else getattr(row, col, default)
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
cm = np.nanmedian(X_m1, axis=0); nm = np.isnan(X_m1); X_m1[nm] = np.take(cm, np.where(nm)[1])
print(f"  M1 matrix: {X_m1.shape}")
gc.collect()

# ── Generate M1 probs — FULL production model for ALL rows ────────────────────
print("\n[M1] Generating probabilities with full production M1 (no OOF)...")
m1_probs_all = (0.70 * model_lr_m1.predict_proba(X_m1)[:,1] +
                0.30 * model_xgb_m1.predict_proba(X_m1)[:,1])
m1_train_acc = accuracy_score(y_train, (m1_probs_all[train_idx] > 0.5).astype(int))
m1_test_acc  = accuracy_score(y_test,  (m1_probs_all[test_idx]  > 0.5).astype(int))
print(f"  M1 train acc (full model on train rows): {m1_train_acc:.4f}")
print(f"  M1 test acc  (full model on test rows):  {m1_test_acc:.4f}")
print(f"  Distributional shift: {abs(m1_train_acc-m1_test_acc):.4f} (was 0.066)")
gc.collect()

# ── Build 42-feature M2A dataset ──────────────────────────────────────────────
print("\n[M2A] Building 42-feature dataset...")
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

m2_rows = []
for i, (_, dr) in enumerate(df.iterrows()):
    m1p = float(m1_probs_all[i])   # ← KEY FIX: full model for all rows
    f1n, f2n, vig_ = novig_probs(dr['R_odds'], dr['B_odds'])
    ml_gap_ = m1p - f1n
    f1_dec=implied_prob(dr['r_dec_odds']) or 0.0; f2_dec=implied_prob(dr['b_dec_odds']) or 0.0
    f1_ko=implied_prob(dr['r_ko_odds']) or 0.0;   f2_ko=implied_prob(dr['b_ko_odds']) or 0.0
    f1_sub=implied_prob(dr['r_sub_odds']) or 0.0; f2_sub=implied_prob(dr['b_sub_odds']) or 0.0
    dec_tot=f1_dec+f2_dec; fin_p=1.0-(dec_tot/2.0) if dec_tot>0 else 0.5
    f1_fin=f1_ko+f1_sub; f2_fin=f2_ko+f2_sub
    m2_rows.append([m1p,f1n,f2n,ml_gap_,vig_,f1_dec,f2_dec,f1_dec-f2_dec,
                    f1_ko,f2_ko,f1_ko-f2_ko,f1_sub,f2_sub,f1_sub-f2_sub,
                    fin_p,f1_fin,f2_fin,f1_fin-f2_fin,abs(ml_gap_),abs(f1n-0.5),abs(m1p-0.5),
                    1 if (m1p>0.5)==(f1n>0.5) else 0, ml_gap_*abs(f1n-0.5)])

X2_base = np.array(m2_rows, dtype=float)
cm2=np.nanmedian(X2_base,axis=0); nm2=np.isnan(X2_base); X2_base[nm2]=np.take(cm2,np.where(nm2)[1])

# Step 1: fav/dog profile + tier (tier computed from training rows only)
df_fh = df[['date','R_fighter','B_fighter','R_odds','B_odds','Winner']].copy()
df_fh['f1_won'] = (df_fh['Winner']=='Red').astype(int)
df_sorted = df_fh.sort_values('date').reset_index()
fav_bouts={}; fav_wins={}; dog_bouts={}; dog_wins={}
tier_train_counts={t:0 for t in range(5)}; tier_train_wins={t:0 for t in range(5)}
step1_rows=[None]*len(df)
train_set = set(train_idx.tolist())
for _, row in df_sorted.iterrows():
    orig_i=row['index']; f1=row['R_fighter']; f1_won=row['f1_won']
    f1nv,_,_ = novig_probs(df.loc[orig_i,'R_odds'], df.loc[orig_i,'B_odds'])
    is_fav = 1 if df.loc[orig_i,'R_odds'] < 0 else 0
    t = 0 if f1nv<0.30 else (1 if f1nv<0.45 else (2 if f1nv<0.55 else (3 if f1nv<0.70 else 4)))
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
step1_arr = np.array(step1_rows, dtype=float)
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

# Step 3: weight-class context (use full-model M1 probs)
wc_arr=X_m1[:,feat_idx['weight_class_ord']]
no_rds=df.get('no_of_rounds', pd.Series([3]*len(df))).fillna(3).values
m1_wc_acc={}
for wc_v in np.unique(wc_arr[train_mask]):
    mask = train_mask & (wc_arr==wc_v)
    if mask.sum()>=5:
        m1_wc_acc[int(wc_v)]=accuracy_score(target[mask],(m1_probs_all[mask]>0.5).astype(int))
m1_train_acc_global = accuracy_score(y_train,(m1_probs_all[train_idx]>0.5).astype(int))
step3_rows=[]
for i in range(len(df)):
    wc_v=wc_arr[i]; is_5r=1 if no_rds[i]>=5 else 0
    wca=m1_wc_acc.get(int(wc_v),m1_train_acc_global)
    m1c=abs(X2_base[i,BASE_M2.index('model_confidence')])
    step3_rows.append([wc_v/11.0, is_5r, wca-m1_train_acc_global, is_5r*m1c])
step3_arr=np.array(step3_rows,dtype=float)

X2_full = np.hstack([X2_base, step1_final, step2_arr, step3_arr])
FEAT_NAMES_2A = (BASE_M2 +
    ['f1_is_fav','f1_hist_fav_wr','f1_hist_dog_wr','f1_fav_bouts_log','f1_dog_bouts_log','odds_strength','tier_hist_win_rate'] +
    ['ko_style_edge','sub_style_edge','finish_x_model_conf','dec_x_str_def',
     'combined_ko_implied','combined_sub_implied','ko_method_gap','sub_method_gap'] +
    ['wc_norm','is_5r','m1_wc_bias','five_r_x_conf'])
assert X2_full.shape[1] == 42
X2_train=X2_full[train_idx]; X2_test=X2_full[test_idx]
print(f"  M2A feature matrix: {X2_full.shape}")
gc.collect()

# ── Train M2A ─────────────────────────────────────────────────────────────────
print("\n[M2A] Training LR + XGB (50/50 blend)...")

new_lr = Pipeline([('sc',RobustScaler()),
                   ('clf',LogisticRegression(C=0.292291,penalty='l2',solver='lbfgs',
                                             max_iter=2000,random_state=SEED))])
new_lr.fit(X2_train, y_train)
lr_cv = cross_val_score(new_lr, X2_train, y_train,
                        cv=StratifiedKFold(5,shuffle=True,random_state=SEED), n_jobs=1).mean()
print(f"  LR CV acc: {lr_cv:.4f}")
gc.collect()

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators',100,400),
        'max_depth':        trial.suggest_int('max_depth',2,6),
        'learning_rate':    trial.suggest_float('learning_rate',0.01,0.2,log=True),
        'subsample':        trial.suggest_float('subsample',0.6,1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.6,1.0),
        'min_child_weight': trial.suggest_int('min_child_weight',1,5),
        'gamma':            trial.suggest_float('gamma',0.0,0.5),
        'reg_lambda':       trial.suggest_float('reg_lambda',0.5,3.0),
    }
    clf = XGBClassifier(**params, eval_metric='logloss', random_state=SEED, verbosity=0, n_jobs=1)
    return cross_val_score(clf, X2_train, y_train,
                           cv=StratifiedKFold(5,shuffle=True,random_state=SEED), n_jobs=1).mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(xgb_objective, n_trials=15, show_progress_bar=False)
print(f"  XGB best CV acc: {study.best_value:.4f}")
new_xgb = XGBClassifier(**study.best_params, eval_metric='logloss', random_state=SEED, verbosity=0, n_jobs=1)
new_xgb.fit(X2_train, y_train)
gc.collect()

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[M2A] Test set evaluation (2024+ men's)...")
lr_p  = new_lr.predict_proba(X2_test)[:,1]
xgb_p = new_xgb.predict_proba(X2_test)[:,1]
m2a_probs = 0.50*lr_p + 0.50*xgb_p
m2a_acc   = accuracy_score(y_test, (m2a_probs>0.5).astype(int))
print(f"  M1 test acc:       {m1_test_acc:.4f}")
print(f"  New M2A test acc:  {m2a_acc:.4f}")
print(f"  Old M2 production: {OLD_M2_ACC:.4f}")
delta = m2a_acc - OLD_M2_ACC
print(f"  Delta vs old:      {delta:+.4f}")

# Per-year breakdown
print("\n  Per-year accuracy (test set):")
years = df['date'].dt.year.values
print(f"  {'Year':>6} {'N':>6} {'M1':>8} {'M2A':>8}")
for yr in sorted(set(years[test_idx])):
    mask = years[test_idx] == yr
    n = mask.sum()
    if n < 5: continue
    ym1 = accuracy_score(target[test_idx][mask], (m1_probs_all[test_idx][mask]>0.5).astype(int))
    ym2 = accuracy_score(target[test_idx][mask], (m2a_probs[mask]>0.5).astype(int))
    print(f"  {yr:>6} {n:>6} {ym1:>8.3f} {ym2:>8.3f}")

# ROI analysis
print("\n  ROI analysis (corrected: odds/100 for underdogs, 100/|odds| for favorites):")
f1_odds_arr = df['R_odds'].values.astype(float)
f2_odds_arr = df['B_odds'].values.astype(float)

def unit_return(odds):
    return 100.0/abs(odds) if odds < 0 else odds/100.0

def roi_sim(probs, threshold, use_agreement=False):
    profits=[]
    for k,(row_i,m2p) in enumerate(zip(test_idx,probs)):
        f1n,f2n,_=novig_probs(f1_odds_arr[row_i],f2_odds_arr[row_i])
        gap=m2p-f1n; m1p_=float(m1_probs_all[row_i])
        if use_agreement and (m2p>0.5)!=(m1p_>0.5): continue
        if abs(gap)<threshold: continue
        bet_f1=gap>0; won=bool(target[row_i]==1) if bet_f1 else bool(target[row_i]==0)
        odds=f1_odds_arr[row_i] if bet_f1 else f2_odds_arr[row_i]
        profits.append(unit_return(float(odds)) if won else -1.0)
    n=len(profits)
    if n==0: return 0,0.0,0.0,0.0
    wins=sum(1 for p in profits if p>0)
    return n,wins/n,sum(profits)/n*100,sum(profits)

THRESHOLDS=[0.05,0.08,0.10,0.12,0.15]
print(f"\n  No filter:   {'Thresh':>8} {'N':>6} {'Win%':>7} {'ROI%':>7}")
std_res={}
for t in THRESHOLDS:
    n,wr,roi,profit=roi_sim(m2a_probs,t,False)
    std_res[t]={'n_bets':n,'win_rate':round(wr,4),'roi_pct':round(roi,2),'total_profit':round(profit,2)}
    print(f"  {'':11}{t:.2f} {n:>6} {wr:>7.3f} {roi:>7.2f}%")

print(f"\n  M1+M2A agree: {'Thresh':>8} {'N':>6} {'Win%':>7} {'ROI%':>7}")
agree_res={}
for t in THRESHOLDS:
    n,wr,roi,profit=roi_sim(m2a_probs,t,True)
    agree_res[t]={'n_bets':n,'win_rate':round(wr,4),'roi_pct':round(roi,2),'total_profit':round(profit,2)}
    print(f"  {'':11}{t:.2f} {n:>6} {wr:>7.3f} {roi:>7.2f}%")

r10_agree_wr = agree_res.get(0.10,{}).get('win_rate',0.0)

# ── Promotion decision ────────────────────────────────────────────────────────
gate1 = m2a_acc > OLD_M2_ACC
gate2 = r10_agree_wr > AGREE_GATE
promotes = gate1 or gate2

print("\n" + "=" * 65)
print(f"Promotion gates:")
print(f"  Gate 1 — accuracy > {OLD_M2_ACC:.4f}: {m2a_acc:.4f} → {'PASS' if gate1 else 'FAIL'}")
print(f"  Gate 2 — 10% agree win rate > {AGREE_GATE:.4f}: {r10_agree_wr:.4f} → {'PASS' if gate2 else 'FAIL'}")

if promotes:
    print(f"\nPROMOTING new M2A (gate {'1' if gate1 else '2'} passed)")
    joblib.dump(new_lr,          'model/ufc_model2a_best.pkl')
    joblib.dump(new_xgb,         'model/ufc_model2a_xgb.pkl')
    joblib.dump(FEAT_NAMES_2A,   'model/ufc_model2a_features.pkl')
    new_tier = {
        'tier_win_rates': {str(k): tier_wr_map[k] for k in sorted(tier_wr_map)},
        'm1_train_acc':   round(m1_train_acc_global,4),
        'm1_wc_acc':      {str(float(k)): round(v,4) for k,v in m1_wc_acc.items()},
        'feature_names':  FEAT_NAMES_2A, 'blend_lr':0.5,'blend_xgb':0.5,
        'test_acc':       round(m2a_acc,4),'n_features':42,
    }
    with open('model/model2a_tier_stats.json','w') as f: json.dump(new_tier,f,indent=2)
    print("  Saved: ufc_model2a_best.pkl, ufc_model2a_xgb.pkl, ufc_model2a_features.pkl, model2a_tier_stats.json")
else:
    print(f"\nHOLDING — neither gate passed. Renaming current M2 files to 2A naming.")
    import shutil
    shutil.copy('model/ufc_model2_best.pkl',    'model/ufc_model2a_best.pkl')
    shutil.copy('model/ufc_model2_xgb.pkl',     'model/ufc_model2a_xgb.pkl')
    shutil.copy('model/ufc_model2_features.pkl','model/ufc_model2a_features.pkl')
    shutil.copy('model/model2_tier_stats.json', 'model/model2a_tier_stats.json')
    print("  Copied: model2_best → model2a_best, model2_xgb → model2a_xgb, etc.")

# Update model_metadata.json
with open('model/model_metadata.json') as f: meta=json.load(f)
meta.setdefault('model2a', {})
meta['model2a'].update({
    'model_type': 'M2A_blend_LR50_XGB50',
    'temporal_accuracy': round(m2a_acc if promotes else OLD_M2_ACC, 4),
    'n_features': 42, 'blend_ratio': '50% LR (l2) + 50% XGB',
    'training_universe': '2018+ men fights with all ML + method odds',
    'n_train': len(train_idx), 'n_test': len(test_idx),
    'date_trained': '2026-05-12',
    'm1_signal': 'full production M1 (no OOF), eliminates distributional shift',
    'promoted': promotes,
    'new_acc': round(m2a_acc,4), 'old_acc': OLD_M2_ACC,
    'prev_attempt_acc': 0.7064,
    'note': 'Fixed OOF distributional shift from previous retrain attempt',
})
with open('model/model_metadata.json','w') as f: json.dump(meta,f,indent=2)

# Save full results
results = {
    'new_m2a_acc': round(m2a_acc,4), 'm1_test_acc': round(m1_test_acc,4),
    'old_m2_acc': OLD_M2_ACC, 'delta': round(delta,4), 'promoted': promotes,
    'n_train': len(train_idx), 'n_test': len(test_idx),
    'm1_signal': 'full_production_no_oof',
    'm1_train_acc_on_train_rows': round(m1_train_acc,4),
    'distributional_shift_pp': round(abs(m1_train_acc-m1_test_acc)*100,2),
    'standard_thresholds': {str(k):v for k,v in std_res.items()},
    'agreement_thresholds': {str(k):v for k,v in agree_res.items()},
    'tier_win_rates': {str(k):round(v,4) for k,v in tier_wr_map.items()},
    'xgb_best_params': study.best_params,
}
with open(RESULTS_DIR/'model2a_retrain_results.json','w') as f: json.dump(results,f,indent=2)
print(f"\nSaved: experiments/research/model2_v2/model2a_retrain_results.json")
print("=" * 65)
print("TASK 1 DONE")
