#!/usr/bin/env python3
"""
8SI UFC Predictor — Architecture Overhaul Sprint
Experiments 1-6: Glicko-2, SHAP reduction, LR-only, alt ensembles,
stacking, combined best. All vs baseline ~72.81% temporal accuracy.

Output piped to architecture_sprint.log
n_jobs=1 throughout. gc.collect() between experiments.
"""
import gc, json, math, os, sys, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.chdir('/Users/allenthompson/Desktop/ufc-predictor')

SEED         = 42
TRAIN_START  = '2015-01-01'
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
HL_DAYS      = 730
N_OPT        = 30          # Optuna trials per model (max 50, using 30 for speed)
CV_FOLDS     = 5
WOMENS = ["Women's Strawweight","Women's Flyweight",
          "Women's Bantamweight","Women's Featherweight"]

RESULTS = []  # accumulated experiment results

def ts():
    return time.strftime('%H:%M:%S')

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}  [{ts()}]")
    print(f"{'='*70}")

def summary_block(name, feat_count, rating, acc, ll, brier, note=""):
    line = (f"\n  ┌{'─'*64}┐\n"
            f"  │  Experiment : {name:<48}│\n"
            f"  │  Features   : {feat_count:<48}│\n"
            f"  │  Rating sys : {rating:<48}│\n"
            f"  │  Test acc   : {acc*100:.4f}%{'':<40}│\n"
            f"  │  Log-loss   : {ll:.6f}{'':<45}│\n"
            f"  │  Brier      : {brier:.6f}{'':<45}│\n"
            f"  │  Note       : {note:<48}│\n"
            f"  └{'─'*64}┘")
    print(line)
    RESULTS.append(dict(name=name, feat_count=feat_count, rating=rating,
                        acc=acc, log_loss=ll, brier=brier, note=note))

def eval_blend(p_lr, p_xgb, y_test, lr_w=0.70, xgb_w=0.30):
    p = lr_w * p_lr + xgb_w * p_xgb
    acc = accuracy_score(y_test, (p[:,1] > 0.5).astype(int))
    ll  = log_loss(y_test, p)
    br  = brier_score_loss(y_test, p[:,1])
    return acc, ll, br, p

def eval_single(p, y_test):
    acc = accuracy_score(y_test, (p[:,1] > 0.5).astype(int))
    ll  = log_loss(y_test, p)
    br  = brier_score_loss(y_test, p[:,1])
    return acc, ll, br

# ─── Helper: corner flip (handles sample weights) ─────────────────────────────
def corner_flip(X, y, w=None):
    Xf = X.copy()
    for col in list(X.columns):
        if col.startswith('R_'):
            b_col = 'B_' + col[2:]
            if b_col in X.columns:
                Xf[col]   = X[b_col].values
                Xf[b_col] = X[col].values
    for col in Xf.columns:
        if col.endswith('_dif'):
            Xf[col] = -Xf[col]
    X_aug = pd.concat([X, Xf], ignore_index=True)
    y_aug = pd.concat([y, 1 - y], ignore_index=True)
    if w is not None:
        w_aug = pd.concat([w, w], ignore_index=True)
        return X_aug, y_aug, w_aug
    return X_aug, y_aug

def compute_weights(dates, cutoff=TRAIN_CUTOFF, hl=HL_DAYS):
    days_before = (cutoff - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_before / hl)

# ─── LR + XGB train-eval helper ───────────────────────────────────────────────
XGB_PARAMS = {'n_estimators':200,'learning_rate':0.1,'max_depth':6,
              'min_child_weight':1,'subsample':0.8,'colsample_bytree':0.7,
              'gamma':0.3,'reg_alpha':0,'reg_lambda':2.0,
              'random_state':SEED,'eval_metric':'logloss','verbosity':0,'n_jobs':1}

LR_C = 0.00711

def train_baseline_models(X_tr_aug, y_tr_aug, w_arr, X_te, y_te,
                           lr_c=LR_C, xgb_p=None):
    if xgb_p is None: xgb_p = XGB_PARAMS
    w_arr_np = w_arr.values if hasattr(w_arr, 'values') else w_arr
    lr_model = Pipeline([('sc', RobustScaler()),
                         ('lr', LogisticRegression(penalty='l2', C=lr_c,
                                 solver='liblinear', max_iter=2000, random_state=SEED))])
    lr_model.fit(X_tr_aug, y_tr_aug, lr__sample_weight=w_arr_np)
    xgb_model = XGBClassifier(**xgb_p)
    xgb_model.fit(X_tr_aug, y_tr_aug, sample_weight=w_arr_np)
    p_lr  = lr_model.predict_proba(X_te)
    p_xgb = xgb_model.predict_proba(X_te)
    return lr_model, xgb_model, p_lr, p_xgb

# ─────────────────────────────────────────────────────────────────────────────
print_section("SHARED DATA LOADING & FEATURE ENGINEERING")
# ─────────────────────────────────────────────────────────────────────────────

print(f"[{ts()}] Loading data...")
df_all = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_all['date'] = pd.to_datetime(df_all['date'])

career_raw = pd.read_csv('data/career_fights_updated.csv')
career_raw['date'] = pd.to_datetime(career_raw['date'])
career_raw = career_raw.sort_values(['fighter','date']).reset_index(drop=True)

style_df = pd.read_csv('data/ufc_fighters_final_updated.csv')
for col in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    style_df[col] = pd.to_numeric(
        style_df[col].astype(str).str.replace('%','',regex=False),
        errors='coerce').fillna(0.0) / 100.0

print(f"  ufc-master: {len(df_all):,} | career: {len(career_raw):,}")

# ── Elo (baseline, K=48) ──────────────────────────────────────────────────────
print(f"[{ts()}] Computing Elo K=48...")
def compute_elo_hist(df_src, K=48, base=1500.0):
    ds = df_src.sort_values('date').reset_index(drop=True)
    elo = {}; rows = []
    for _, row in ds.iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        rb, bb = elo.get(r, base), elo.get(b, base)
        re = 1.0/(1.0+10.0**((bb-rb)/400.0))
        ra = 1.0 if row['Winner']=='Red' else (0.0 if row['Winner']=='Blue' else 0.5)
        ba = 1.0 - ra
        ra_new, ba_new = rb + K*(ra-re), bb + K*(ba-(1-re))
        rows += [{'fighter':r,'date':row['date'],'elo_before':rb,'elo_after':ra_new},
                 {'fighter':b,'date':row['date'],'elo_before':bb,'elo_after':ba_new}]
        elo[r], elo[b] = ra_new, ba_new
    h = pd.DataFrame(rows).sort_values(['fighter','date']).reset_index(drop=True)
    h['elo_trend'] = h.groupby('fighter')['elo_before'].transform(lambda x: x - x.shift(3))
    return h

elo_hist = compute_elo_hist(df_all)
print(f"  Elo history: {len(elo_hist):,} rows")

# ── Career stats ──────────────────────────────────────────────────────────────
print(f"[{ts()}] Computing career stats (shift=1)...")
def compute_career_stats(cdf):
    df = cdf.sort_values(['fighter','date']).copy().reset_index(drop=True)
    df['_ko']  = ((df['won']==1) & df['method'].str.contains('KO|TKO',case=False,na=False)).astype(float)
    df['_sub'] = ((df['won']==1) & df['method'].str.contains('Sub|Submission',case=False,na=False)).astype(float)
    df['_fin'] = ((df['won']==1) & df['method'].str.contains('KO|TKO|Sub',case=False,na=False)).astype(float)
    g = df.groupby('fighter', sort=False)
    df['cum_fights'] = g.cumcount()
    for src, dst in [('won','_cs_won'),('_ko','_cs_ko'),('_sub','_cs_sub'),('_fin','_cs_fin')]:
        df[dst] = g[src].cumsum() - df[src]
    sn = df['cum_fights'].clip(lower=1)
    df['career_win_rate']    = np.where(df['cum_fights']>0, df['_cs_won']/sn, 0.5)
    df['ko_finish_rate']     = np.where(df['cum_fights']>0, df['_cs_ko']/sn,  0.0)
    df['sub_finish_rate']    = np.where(df['cum_fights']>0, df['_cs_sub']/sn, 0.0)
    df['career_finish_rate'] = np.where(df['cum_fights']>0, df['_cs_fin']/sn, 0.0)
    def _roll(s, w, d): return s.shift(1).rolling(w, min_periods=1).mean().fillna(d)
    df['last3_win_rate']    = g['won'].transform(lambda x: _roll(x,3,0.5))
    df['last10_win_rate']   = g['won'].transform(lambda x: _roll(x,10,0.5))
    df['last5_won']         = g['won'].transform(lambda x: _roll(x,5,0.5))
    df['last5_finish_rate'] = g['_fin'].transform(lambda x: _roll(x,5,0.0))
    df['trend_score']       = df['last3_win_rate'] - df['last10_win_rate']
    df['_prev_date']        = g['date'].transform(lambda x: x.shift(1))
    df['layoff_days']       = (df['date']-df['_prev_date']).dt.days.fillna(180.0).clip(lower=0)
    all_wr = {f: grp['won'].sum()/max(1,len(grp)) for f,grp in df.groupby('fighter')}
    opp_col = df['opponent'].tolist(); ftr_col = df['fighter'].tolist()
    ftr_pos = defaultdict(list)
    for pos, idx in enumerate(df.index.tolist()):
        ftr_pos[ftr_col[pos]].append(pos)
    oq = np.full(len(df), 0.5)
    for ftr, positions in ftr_pos.items():
        for rank, pos in enumerate(positions):
            past = [opp_col[p] for p in positions[max(0,rank-5):rank]]
            rates = [all_wr.get(o,0.5) for o in past]
            oq[pos] = float(np.mean(rates)) if rates else 0.5
    df['opp_quality'] = oq
    keep = ['fighter','date','cum_fights','career_win_rate','ko_finish_rate',
            'sub_finish_rate','career_finish_rate','last3_win_rate','last10_win_rate',
            'last5_won','last5_finish_rate','trend_score','layoff_days','opp_quality']
    return df[keep]

career_stats = compute_career_stats(career_raw)
print(f"  Career rows: {len(career_stats):,}")

# ── QA stats ──────────────────────────────────────────────────────────────────
print(f"[{ts()}] Computing QA stats...")
def compute_qa_stats(cdf, elo_h):
    elo_ref = elo_h[['fighter','date','elo_before']].copy()
    od = cdf[['fighter','opponent','date','won','got_finish']].copy()
    od = od.rename(columns={'opponent':'opp_name'})
    oref = elo_ref.rename(columns={'fighter':'opp_name','elo_before':'opp_elo'})
    od = pd.merge_asof(od.sort_values('date'), oref.sort_values('date'),
                       on='date', by='opp_name', direction='backward')
    od['opp_elo'] = od['opp_elo'].fillna(1500.0)
    od['ew']      = od['opp_elo'] / 1500.0
    rows = []
    for fighter, grp in od.groupby('fighter', sort=False):
        grp = grp.sort_values('date'); n = len(grp)
        qa_wr = np.full(n, 0.5); qa_fr = np.full(n, 0.0)
        qa_sl = np.full(n, 0.0); qa_sa = np.full(n, 0.0)
        cew = ceww = cewf = cn = coff = cdef_ = 0.0
        for i, (_, r) in enumerate(grp.iterrows()):
            if cew>0: qa_wr[i] = ceww/cew; qa_fr[i] = cewf/cew
            if cn>0:  qa_sl[i] = coff/cn;  qa_sa[i] = cdef_/cn
            ew=r['ew']; w=r['won']; f=r['got_finish'] if pd.notna(r.get('got_finish')) else 0.0
            cew+=ew; ceww+=ew*w; cewf+=ew*f; cn+=ew; coff+=ew*w; cdef_+=ew*(1.0-w)
        rows.append(pd.DataFrame({'fighter':fighter,'date':grp['date'].values,
                                   'qa_win_rate':qa_wr,'qa_finish_rate':qa_fr,
                                   'qa_SLpM':qa_sl,'qa_SApM':qa_sa}))
    return pd.concat(rows, ignore_index=True).sort_values(['fighter','date'])

qa_stats = compute_qa_stats(career_raw, elo_hist)
print(f"  QA rows: {len(qa_stats):,}")

# ── Build main fight DataFrame (men's, 2015+) ─────────────────────────────────
print(f"[{ts()}] Building fight DataFrame...")
WC_ORDER = {"Women's Strawweight":0,"Women's Flyweight":1,"Women's Bantamweight":2,
            "Women's Featherweight":3,"Flyweight":4,"Bantamweight":5,
            "Featherweight":6,"Lightweight":7,"Welterweight":8,
            "Middleweight":9,"Light Heavyweight":10,"Heavyweight":11,"Catch Weight":6}

df = df_all[(df_all['date'] >= TRAIN_START) & df_all['Winner'].isin(['Red','Blue']) &
            ~df_all['weight_class'].isin(WOMENS)].copy().sort_values('date').reset_index(drop=True)
print(f"  Fights after filter: {len(df):,}")

# Merge career stats
cc = [c for c in career_stats.columns if c not in ('fighter','date')]
r_cs = career_stats.rename(columns={'fighter':'R_fighter',**{c:f'R_{c}' for c in cc}})
b_cs = career_stats.rename(columns={'fighter':'B_fighter',**{c:f'B_{c}' for c in cc}})
df = pd.merge_asof(df.sort_values('date'), r_cs.sort_values('date'),
                   on='date', by='R_fighter', direction='backward')
df = pd.merge_asof(df.sort_values('date'), b_cs.sort_values('date'),
                   on='date', by='B_fighter', direction='backward')
cdef = {'cum_fights':0,'career_win_rate':0.5,'ko_finish_rate':0.0,'sub_finish_rate':0.0,
        'career_finish_rate':0.0,'last3_win_rate':0.5,'last10_win_rate':0.5,
        'last5_won':0.5,'last5_finish_rate':0.0,'trend_score':0.0,
        'layoff_days':180.0,'opp_quality':0.5}
for stat, dv in cdef.items():
    df[f'R_{stat}'] = df[f'R_{stat}'].fillna(dv)
    df[f'B_{stat}'] = df[f'B_{stat}'].fillna(dv)

# Merge QA stats
qa_r = qa_stats.rename(columns={'fighter':'R_fighter','qa_win_rate':'R_qa_win_rate',
    'qa_finish_rate':'R_qa_finish_rate','qa_SLpM':'R_qa_SLpM','qa_SApM':'R_qa_SApM'})
qa_b = qa_stats.rename(columns={'fighter':'B_fighter','qa_win_rate':'B_qa_win_rate',
    'qa_finish_rate':'B_qa_finish_rate','qa_SLpM':'B_qa_SLpM','qa_SApM':'B_qa_SApM'})
df = pd.merge_asof(df.sort_values('date'), qa_r.sort_values('date'),
                   on='date', by='R_fighter', direction='backward')
df = pd.merge_asof(df.sort_values('date'), qa_b.sort_values('date'),
                   on='date', by='B_fighter', direction='backward')
for c in ['R_qa_win_rate','R_qa_finish_rate','R_qa_SLpM','R_qa_SApM',
          'B_qa_win_rate','B_qa_finish_rate','B_qa_SLpM','B_qa_SApM']:
    df[c] = df[c].fillna(0.5 if 'win_rate' in c else 0.0)
df['qa_win_rate_dif']    = df['R_qa_win_rate']    - df['B_qa_win_rate']
df['qa_finish_rate_dif'] = df['R_qa_finish_rate'] - df['B_qa_finish_rate']
df['qa_SLpM_dif']        = df['R_qa_SLpM']        - df['B_qa_SLpM']
df['qa_SApM_dif']        = df['R_qa_SApM']        - df['B_qa_SApM']

# Merge Elo
er = elo_hist[['fighter','date','elo_before','elo_trend']].rename(
    columns={'fighter':'R_fighter','elo_before':'R_elo','elo_trend':'R_elo_trend'})
eb = elo_hist[['fighter','date','elo_before','elo_trend']].rename(
    columns={'fighter':'B_fighter','elo_before':'B_elo','elo_trend':'B_elo_trend'})
df = pd.merge_asof(df.sort_values('date'), er.sort_values('date'),
                   on='date', by='R_fighter', direction='backward')
df = pd.merge_asof(df.sort_values('date'), eb.sort_values('date'),
                   on='date', by='B_fighter', direction='backward')
for c in ['R_elo','B_elo']: df[c] = df[c].fillna(1500.0)
for c in ['R_elo_trend','B_elo_trend']: df[c] = df[c].fillna(0.0)
df['elo_dif']       = df['R_elo']       - df['B_elo']
df['elo_trend_dif'] = df['R_elo_trend'] - df['B_elo_trend']

# Merge style stats
style_src = ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']
sdf = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last')
rs = sdf[['Fighter_Name']+style_src].rename(
    columns={'Fighter_Name':'R_fighter',**{c:f'R_{c}' for c in style_src}})
bs = sdf[['Fighter_Name']+style_src].rename(
    columns={'Fighter_Name':'B_fighter',**{c:f'B_{c}' for c in style_src}})
df = df.merge(rs, on='R_fighter', how='left').merge(bs, on='B_fighter', how='left')
for col in [f'{p}{s}' for p in ('R_','B_') for s in style_src]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# Feature engineering
df['weight_class_ord'] = df['weight_class'].map(WC_ORDER).fillna(6).astype(int)
df['R_southpaw'] = (df['R_Stance'].str.lower()=='southpaw').astype(int)
df['B_southpaw'] = (df['B_Stance'].str.lower()=='southpaw').astype(int)
df['orth_clash']  = ((df['R_southpaw']==0)&(df['B_southpaw']==0)).astype(int)
df['south_clash'] = ((df['R_southpaw']==1)&(df['B_southpaw']==1)).astype(int)
df['R_age'] = pd.to_numeric(df['R_age'],errors='coerce').fillna(28.0)
df['B_age'] = pd.to_numeric(df['B_age'],errors='coerce').fillna(28.0)
df['R_age_x_exp']   = df['R_age'] * df['R_cum_fights']
df['B_age_x_exp']   = df['B_age'] * df['B_cum_fights']
df['age_x_exp_dif'] = df['R_age_x_exp'] - df['B_age_x_exp']
for p,s in [('R_',df['R_layoff_days']),('B_',df['B_layoff_days'])]:
    d = s.fillna(180.0)
    df[f'{p}layoff_lt90']    = (d<90).astype(int)
    df[f'{p}layoff_90_180']  = ((d>=90)&(d<180)).astype(int)
    df[f'{p}layoff_180_365'] = ((d>=180)&(d<365)).astype(int)
    df[f'{p}layoff_gt365']   = (d>=365).astype(int)
for stat in ['career_win_rate','last5_won','last5_finish_rate','opp_quality',
             'trend_score','ko_finish_rate','sub_finish_rate','last3_win_rate','last10_win_rate']:
    df[f'{stat}_dif'] = df[f'R_{stat}'] - df[f'B_{stat}']
for pair in [('SLpM','SLpM_dif'),('SApM','SApM_dif'),('Str_Def','Str_Def_dif'),
             ('TD_Def','TD_Def_dif'),('Sub_Avg','Sub_Avg_dif'),('TD_Avg','TD_Avg_dif')]:
    df[pair[1]] = df[f'R_{pair[0]}'] - df[f'B_{pair[0]}']
df['win_dif']             = df['R_wins']                - df['B_wins']
df['loss_dif']            = df['R_losses']              - df['B_losses']
df['win_streak_dif']      = df['R_current_win_streak']  - df['B_current_win_streak']
df['lose_streak_dif']     = df['R_current_lose_streak'] - df['B_current_lose_streak']
df['height_dif']          = df['R_Height_cms']          - df['B_Height_cms']
df['reach_dif']           = df['R_Reach_cms']           - df['B_Reach_cms']
df['age_dif']             = df['R_age']                 - df['B_age']
df['sig_str_dif']         = df['R_avg_SIG_STR_landed']  - df['B_avg_SIG_STR_landed']
df['avg_td_dif']          = df['R_avg_TD_landed']       - df['B_avg_TD_landed']
df['ko_dif']              = df['R_ko_finish_rate']      - df['B_ko_finish_rate']
df['sub_dif']             = df['R_sub_finish_rate']     - df['B_sub_finish_rate']
df['total_title_bout_dif']= 0

# Interaction features (got_finished_rate, age_x_layoff, finish_danger_mismatch)
print(f"[{ts()}] Computing interaction features...")
cdf2 = career_raw.sort_values(['fighter','date']).copy()
cdf2['is_loss']     = (cdf2['won']==0).astype(float)
cdf2['is_fin_loss'] = ((cdf2['won']==0)&(cdf2['got_finish'].fillna(0)==1)).astype(float)
g2 = cdf2.groupby('fighter', sort=False)
cdf2['_cs_l']  = g2['is_loss'].cumsum()    - cdf2['is_loss']
cdf2['_cs_fl'] = g2['is_fin_loss'].cumsum()- cdf2['is_fin_loss']
cdf2['got_finished_rate'] = np.where(cdf2['_cs_l']>0, cdf2['_cs_fl']/cdf2['_cs_l'], 0.5)
chin = cdf2[['fighter','date','got_finished_rate']].sort_values(['fighter','date'])
cr2 = chin.rename(columns={'fighter':'R_fighter','got_finished_rate':'R_got_finished_rate'})
cb2 = chin.rename(columns={'fighter':'B_fighter','got_finished_rate':'B_got_finished_rate'})
df = pd.merge_asof(df.sort_values('date'), cr2.sort_values('date'),
                   on='date', by='R_fighter', direction='backward')
df = pd.merge_asof(df.sort_values('date'), cb2.sort_values('date'),
                   on='date', by='B_fighter', direction='backward')
df['R_got_finished_rate'] = df['R_got_finished_rate'].fillna(0.5)
df['B_got_finished_rate'] = df['B_got_finished_rate'].fillna(0.5)
df['R_age_x_layoff']       = df['R_age'] * df['R_layoff_days'].clip(upper=730)
df['B_age_x_layoff']       = df['B_age'] * df['B_layoff_days'].clip(upper=730)
df['age_x_layoff_dif']     = df['R_age_x_layoff'] - df['B_age_x_layoff']
df['R_finish_danger']      = df['R_ko_finish_rate'] + df['R_sub_finish_rate']
df['B_finish_danger']      = df['B_ko_finish_rate'] + df['B_sub_finish_rate']
df['finish_danger_mismatch'] = (df['R_finish_danger'] * (1-df['B_got_finished_rate']) -
                                df['B_finish_danger'] * (1-df['R_got_finished_rate']))
del cdf2, g2, chin, cr2, cb2; gc.collect()

# Debut filter
df = df[(df['R_cum_fights']>=1)&(df['B_cum_fights']>=1)].copy().reset_index(drop=True)
df['target'] = (df['Winner']=='Red').astype(int)
print(f"  After debut filter: {len(df):,}  train={( df['date'] < TRAIN_CUTOFF).sum()}  test={(df['date']>=TRAIN_CUTOFF).sum()}")

# ── Baseline 129-feature list ─────────────────────────────────────────────────
FEAT_BASE = [
    "R_wins","R_losses","R_Height_cms","R_age",
    "R_avg_SIG_STR_landed","R_avg_TD_landed",
    "R_current_win_streak","R_current_lose_streak","R_longest_win_streak",
    "R_avg_SIG_STR_pct","R_avg_SUB_ATT","R_avg_TD_pct","R_Reach_cms",
    "B_wins","B_losses","B_Height_cms","B_age",
    "B_avg_SIG_STR_landed","B_avg_TD_landed",
    "B_current_win_streak","B_current_lose_streak","B_longest_win_streak",
    "B_avg_SIG_STR_pct","B_avg_SUB_ATT","B_avg_TD_pct","B_Reach_cms","B_total_title_bouts",
    "win_dif","loss_dif","win_streak_dif","lose_streak_dif",
    "height_dif","reach_dif","age_dif","sig_str_dif",
    "avg_td_dif","ko_dif","sub_dif","total_title_bout_dif",
    "weight_class_ord","orth_clash","south_clash","R_southpaw",
    "R_cum_fights","B_cum_fights",
    "R_career_win_rate","B_career_win_rate","career_win_rate_dif",
    "R_last5_won","B_last5_won","last5_won_dif",
    "R_last5_finish_rate","B_last5_finish_rate","last5_finish_rate_dif",
    "R_opp_quality","B_opp_quality","opp_quality_dif",
    "R_trend_score","B_trend_score","trend_score_dif",
    "R_ko_finish_rate","B_ko_finish_rate","ko_finish_rate_dif",
    "R_sub_finish_rate","B_sub_finish_rate","sub_finish_rate_dif",
    "R_last3_win_rate","B_last3_win_rate","last3_win_rate_dif",
    "R_last10_win_rate","B_last10_win_rate",
    "R_age_x_exp","B_age_x_exp","age_x_exp_dif",
    "R_layoff_lt90","R_layoff_90_180","R_layoff_180_365","R_layoff_gt365",
    "B_layoff_lt90","B_layoff_90_180","B_layoff_180_365",
    "R_SLpM","R_SApM","R_Str_Acc","R_Str_Def","R_TD_Avg","R_TD_Acc","R_TD_Def","R_Sub_Avg",
    "B_SLpM","B_SApM","B_Str_Acc","B_Str_Def","B_TD_Avg","B_TD_Acc","B_TD_Def","B_Sub_Avg",
    "SLpM_dif","SApM_dif","Str_Def_dif","TD_Def_dif","Sub_Avg_dif","TD_Avg_dif",
    "R_elo","B_elo","elo_dif","R_elo_trend","B_elo_trend","elo_trend_dif",
]
FEAT_QA  = ["R_qa_win_rate","R_qa_finish_rate","R_qa_SLpM","R_qa_SApM",
            "B_qa_win_rate","B_qa_finish_rate","B_qa_SLpM","B_qa_SApM",
            "qa_win_rate_dif","qa_finish_rate_dif","qa_SLpM_dif","qa_SApM_dif"]
FEAT_INT = ["R_age_x_layoff","B_age_x_layoff","age_x_layoff_dif",
            "R_finish_danger","B_finish_danger","finish_danger_mismatch",
            "R_got_finished_rate","B_got_finished_rate"]
FEAT_129 = FEAT_BASE + FEAT_QA + FEAT_INT

# Force numeric
for col in FEAT_129:
    df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0.0)

# Train / test split
train_mask = df['date'] < TRAIN_CUTOFF
test_mask  = ~train_mask
X_tr_raw  = df.loc[train_mask, FEAT_129].reset_index(drop=True)
y_tr_raw  = df.loc[train_mask, 'target'].reset_index(drop=True)
d_tr_raw  = df.loc[train_mask, 'date'].reset_index(drop=True)
X_te      = df.loc[test_mask,  FEAT_129].reset_index(drop=True)
y_te      = df.loc[test_mask,  'target'].reset_index(drop=True)

w_raw    = pd.Series(compute_weights(d_tr_raw), index=y_tr_raw.index)
X_tr_aug, y_tr_aug, w_tr_aug = corner_flip(X_tr_raw, y_tr_raw, w_raw)
w_arr = w_tr_aug.values

print(f"  Train (augmented): {len(X_tr_aug):,}  Test: {len(X_te):,}  Features: {len(FEAT_129)}")
print(f"[{ts()}] Shared setup complete. Starting experiments...\n")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("BASELINE — LR 70% + XGB 30%, 129 features, Elo K=48")
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
lr_base, xgb_base, p_lr_b, p_xgb_b = train_baseline_models(
    X_tr_aug, y_tr_aug, w_arr, X_te, y_te)
acc_b, ll_b, br_b, p_blend_b = eval_blend(p_lr_b, p_xgb_b, y_te)
print(f"  Baseline: acc={acc_b*100:.4f}%  ll={ll_b:.6f}  brier={br_b:.6f}  [{time.time()-t0:.1f}s]")
summary_block("BASELINE (LR70+XGB30, Elo, 129 feats)",129,"Elo K=48",
              acc_b, ll_b, br_b, f"Reference: {acc_b*100:.2f}%")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 1 — Glicko-2 (standard + domain-separated)")
# ══════════════════════════════════════════════════════════════════════════════

def glicko2_update(mu, phi, sigma, mu_j, phi_j, s, tau=0.5):
    """Single-game Glicko-2 update. All values in Glicko-2 scale."""
    g_j = 1.0 / math.sqrt(1.0 + 3.0 * phi_j**2 / math.pi**2)
    E   = 1.0 / (1.0 + math.exp(-g_j * (mu - mu_j)))
    v   = 1.0 / (g_j**2 * E * (1.0 - E) + 1e-10)
    delta = v * g_j * (s - E)
    a = math.log(max(sigma**2, 1e-15))
    def f(x):
        ex = math.exp(x)
        d2 = phi**2 + v + ex
        return (ex*(delta**2 - phi**2 - v - ex)/(2.0*d2**2+1e-15)) - (x-a)/tau**2
    A = a
    B = math.log(max(delta**2 - phi**2 - v, 1e-15)) if delta**2 > phi**2 + v else a - tau
    fa, fb = f(A), f(B)
    for _ in range(50):
        if abs(B - A) < 1e-6: break
        C = A + (A - B) * fa / (fb - fa + 1e-15)
        fc = f(C)
        if fc * fb < 0: A, fa = B, fb
        else: fa /= 2.0
        B, fb = C, fc
    sigma_new = math.exp(A / 2.0)
    phi_star = math.sqrt(phi**2 + sigma_new**2)
    phi_new  = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
    mu_new   = mu + phi_new**2 * g_j * (s - E)
    return mu_new, phi_new, sigma_new

def compute_glicko2(df_src, tau=0.5, init_r=1500, init_rd=350, init_s=0.06):
    SCALE = 173.7178
    state = {}; rows = []
    for _, row in df_src.sort_values('date').iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        rm, rp, rs = state.get(r, ((init_r-1500)/SCALE, init_rd/SCALE, init_s))
        bm, bp, bs = state.get(b, ((init_r-1500)/SCALE, init_rd/SCALE, init_s))
        ra = 1.0 if row['Winner']=='Red' else (0.0 if row['Winner']=='Blue' else 0.5)
        ba = 1.0 - ra
        rows += [{'fighter':r,'date':row['date'],
                  'g2_r':rm*SCALE+1500,'g2_rd':rp*SCALE,'g2_sigma':rs},
                 {'fighter':b,'date':row['date'],
                  'g2_r':bm*SCALE+1500,'g2_rd':bp*SCALE,'g2_sigma':bs}]
        rm_new,rp_new,rs_new = glicko2_update(rm,rp,rs,bm,bp,ra,tau)
        bm_new,bp_new,bs_new = glicko2_update(bm,bp,bs,rm,rp,ba,tau)
        state[r] = (rm_new,rp_new,rs_new)
        state[b] = (bm_new,bp_new,bs_new)
    return pd.DataFrame(rows)

print(f"[{ts()}] Computing Glicko-2 standard (tau=0.5)...")
g2_hist = compute_glicko2(df_all)
print(f"  G2 history: {len(g2_hist):,} rows")

def merge_g2(df_fights, g2_df, r_prefix='R_', b_prefix='B_'):
    gr = g2_df.rename(columns={'fighter':f'{r_prefix}fighter','g2_r':f'{r_prefix}g2_r',
                                 'g2_rd':f'{r_prefix}g2_rd','g2_sigma':f'{r_prefix}g2_sigma'})
    gb = g2_df.rename(columns={'fighter':f'{b_prefix}fighter','g2_r':f'{b_prefix}g2_r',
                                 'g2_rd':f'{b_prefix}g2_rd','g2_sigma':f'{b_prefix}g2_sigma'})
    out = pd.merge_asof(df_fights.sort_values('date'), gr.sort_values('date'),
                        on='date', by=f'{r_prefix}fighter', direction='backward')
    out = pd.merge_asof(out.sort_values('date'), gb.sort_values('date'),
                        on='date', by=f'{b_prefix}fighter', direction='backward')
    for c in [f'{r_prefix}g2_r',f'{b_prefix}g2_r']:   out[c] = out[c].fillna(1500.0)
    for c in [f'{r_prefix}g2_rd',f'{b_prefix}g2_rd']: out[c] = out[c].fillna(350.0)
    for c in [f'{r_prefix}g2_sigma',f'{b_prefix}g2_sigma']: out[c] = out[c].fillna(0.06)
    out['g2_r_dif']     = out[f'{r_prefix}g2_r']     - out[f'{b_prefix}g2_r']
    out['g2_rd_dif']    = out[f'{r_prefix}g2_rd']    - out[f'{b_prefix}g2_rd']
    out['g2_sigma_dif'] = out[f'{r_prefix}g2_sigma'] - out[f'{b_prefix}g2_sigma']
    return out

df_g2 = merge_g2(df.copy(), g2_hist)
G2_FEATS    = ['R_g2_r','B_g2_r','g2_r_dif','R_g2_rd','B_g2_rd','g2_rd_dif']
G2_ALL_FEATS= ['R_g2_r','B_g2_r','g2_r_dif','R_g2_rd','B_g2_rd','g2_rd_dif',
               'R_g2_sigma','B_g2_sigma','g2_sigma_dif']

# Replace Elo features with Glicko-2 (same count)
ELO_FEATS = ['R_elo','B_elo','elo_dif','R_elo_trend','B_elo_trend','elo_trend_dif']
FEAT_G2_STD = [f for f in FEAT_129 if f not in ELO_FEATS] + G2_FEATS

# Force G2 features numeric
for col in G2_ALL_FEATS:
    df_g2[col] = pd.to_numeric(df_g2.get(col, 0), errors='coerce').fillna(
        1500.0 if '_r' in col and 'rd' not in col and 'sigma' not in col else
        (350.0 if '_rd' in col else 0.06 if 'sigma' in col else 0.0))

X_tr_g2 = df_g2.loc[train_mask, FEAT_G2_STD].reset_index(drop=True)
X_te_g2 = df_g2.loc[test_mask,  FEAT_G2_STD].reset_index(drop=True)

X_tr_g2_aug, y_tr_g2_aug, w_g2_aug = corner_flip(X_tr_g2, y_tr_raw, w_raw)
print(f"[{ts()}] Training Exp1-A (Glicko-2 standard, {len(FEAT_G2_STD)} feats)...")
lr_g2, xgb_g2, plr_g2, pxgb_g2 = train_baseline_models(
    X_tr_g2_aug, y_tr_g2_aug, w_g2_aug.values, X_te_g2, y_te)
acc_g2, ll_g2, br_g2, _ = eval_blend(plr_g2, pxgb_g2, y_te)
print(f"  G2 standard: acc={acc_g2*100:.4f}%  ll={ll_g2:.6f}  brier={br_g2:.6f}")
summary_block("EXP1A: Glicko-2 Standard (replaces Elo)",len(FEAT_G2_STD),
              "Glicko-2 tau=0.5",acc_g2,ll_g2,br_g2,
              f"vs baseline: {(acc_g2-acc_b)*100:+.3f}pp")
gc.collect()

# Domain-separated Glicko-2
print(f"\n[{ts()}] Computing domain-separated Glicko-2...")

def compute_glicko2_domain(df_src, method_filter, tau=0.5, init_r=1500, init_rd=350, init_s=0.06):
    """Glicko-2 updated ONLY for fights matching method_filter (e.g. KO, Sub, Dec)."""
    SCALE = 173.7178
    state = {}; rows = []
    for _, row in df_src.sort_values('date').iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        rm, rp, rs = state.get(r, ((init_r-1500)/SCALE, init_rd/SCALE, init_s))
        bm, bp, bs = state.get(b, ((init_r-1500)/SCALE, init_rd/SCALE, init_s))
        rows += [{'fighter':r,'date':row['date'],'g2_r':rm*SCALE+1500},
                 {'fighter':b,'date':row['date'],'g2_r':bm*SCALE+1500}]
        method = str(row.get('finish','') or '')
        if method_filter.lower() in method.lower():
            ra = 1.0 if row['Winner']=='Red' else (0.0 if row['Winner']=='Blue' else 0.5)
            ba = 1.0 - ra
            rm_new,rp_new,rs_new = glicko2_update(rm,rp,rs,bm,bp,ra,tau)
            bm_new,bp_new,bs_new = glicko2_update(bm,bp,bs,rm,rp,ba,tau)
            state[r] = (rm_new,rp_new,rs_new)
            state[b] = (bm_new,bp_new,bs_new)
    return pd.DataFrame(rows)

g2_strike = compute_glicko2_domain(df_all, 'KO')
g2_grapple = compute_glicko2_domain(df_all, 'Sub')
g2_dec     = compute_glicko2_domain(df_all, 'Decision')

def merge_domain_g2(df_fights, g2_df, domain):
    gr = g2_df.rename(columns={'fighter':f'R_fighter','g2_r':f'R_g2_{domain}'})
    gb = g2_df.rename(columns={'fighter':f'B_fighter','g2_r':f'B_g2_{domain}'})
    out = pd.merge_asof(df_fights.sort_values('date'), gr.sort_values('date'),
                        on='date', by='R_fighter', direction='backward')
    out = pd.merge_asof(out.sort_values('date'), gb.sort_values('date'),
                        on='date', by='B_fighter', direction='backward')
    out[f'R_g2_{domain}'] = out[f'R_g2_{domain}'].fillna(1500.0)
    out[f'B_g2_{domain}'] = out[f'B_g2_{domain}'].fillna(1500.0)
    out[f'g2_{domain}_dif'] = out[f'R_g2_{domain}'] - out[f'B_g2_{domain}']
    return out

df_g2d = df_g2.copy()
df_g2d = merge_domain_g2(df_g2d, g2_strike,  'strike')
df_g2d = merge_domain_g2(df_g2d, g2_grapple, 'grapple')
df_g2d = merge_domain_g2(df_g2d, g2_dec,     'dec')
del g2_strike, g2_grapple, g2_dec; gc.collect()

DOMAIN_FEATS = ['g2_strike_dif','g2_grapple_dif','g2_dec_dif']
FEAT_G2_DOM  = FEAT_G2_STD + DOMAIN_FEATS

X_tr_gd = df_g2d.loc[train_mask, FEAT_G2_DOM].reset_index(drop=True)
X_te_gd = df_g2d.loc[test_mask,  FEAT_G2_DOM].reset_index(drop=True)
X_tr_gd_aug, y_gd_aug, w_gd_aug = corner_flip(X_tr_gd, y_tr_raw, w_raw)
print(f"[{ts()}] Training Exp1-B (Glicko-2 domain, {len(FEAT_G2_DOM)} feats)...")
lr_gd, xgb_gd, plr_gd, pxgb_gd = train_baseline_models(
    X_tr_gd_aug, y_gd_aug, w_gd_aug.values, X_te_gd, y_te)
acc_gd, ll_gd, br_gd, _ = eval_blend(plr_gd, pxgb_gd, y_te)
print(f"  G2 domain: acc={acc_gd*100:.4f}%  ll={ll_gd:.6f}  brier={br_gd:.6f}")
summary_block("EXP1B: Glicko-2 + Domain (striking/grappling/dec)",len(FEAT_G2_DOM),
              "Glicko-2 standard + 3 domain",acc_gd,ll_gd,br_gd,
              f"vs baseline: {(acc_gd-acc_b)*100:+.3f}pp")
del df_g2, df_g2d; gc.collect()

# Best Glicko-2 for later use
best_g2_acc  = acc_g2 if acc_g2 >= acc_gd else acc_gd
best_g2_feat = FEAT_G2_STD if acc_g2 >= acc_gd else FEAT_G2_DOM
best_g2_name = "G2-standard" if acc_g2 >= acc_gd else "G2-domain"
print(f"\n  Best Glicko-2 variant: {best_g2_name}")

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 2 — SHAP Feature Reduction")
# ══════════════════════════════════════════════════════════════════════════════

print(f"[{ts()}] Running SHAP TreeExplainer on baseline XGBoost ({len(FEAT_129)} features)...")
# Use 400 training examples for SHAP (fast on TreeExplainer)
rng = np.random.RandomState(SEED)
shap_idx = rng.choice(len(X_tr_aug), min(400, len(X_tr_aug)), replace=False)
X_shap   = X_tr_aug.iloc[shap_idx]
explainer = shap.TreeExplainer(xgb_base)
shap_vals = explainer.shap_values(X_shap)
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
shap_ranking  = sorted(zip(FEAT_129, mean_abs_shap), key=lambda x: -x[1])
print(f"  Top 15 features by SHAP:")
for i, (feat, sv) in enumerate(shap_ranking[:15], 1):
    print(f"    {i:2d}. {feat:<35}: {sv:.5f}")

# Flag raw stats in top 25
raw_flags = []
for feat, sv in shap_ranking[:25]:
    if feat in ['R_wins','R_losses','B_wins','B_losses',
                'R_Height_cms','R_Reach_cms','B_Height_cms','B_Reach_cms',
                'R_age','B_age','R_avg_SIG_STR_landed','B_avg_SIG_STR_landed',
                'R_avg_TD_landed','B_avg_TD_landed']:
        raw_flags.append(feat)
if raw_flags:
    print(f"\n  Raw stats in top 25 (candidates for ratio replacement): {raw_flags}")

top60 = [f for f,_ in shap_ranking[:60]]
top40 = [f for f,_ in shap_ranking[:40]]
top25 = [f for f,_ in shap_ranking[:25]]

del explainer, shap_vals, X_shap; gc.collect()

exp2_results = {}
for n_feats, feat_list, label in [(60,top60,"Top60"),(40,top40,"Top40"),(25,top25,"Top25")]:
    print(f"\n[{ts()}] Experiment 2 — {label} ({n_feats} features)...")
    X_tr_r = X_tr_aug[feat_list]; X_te_r = X_te[feat_list]

    # LR alone
    w_arr_np = w_tr_aug.values
    lr_r = Pipeline([('sc', RobustScaler()),
                     ('lr', LogisticRegression(penalty='l2',C=LR_C,solver='liblinear',
                                               max_iter=2000,random_state=SEED))])
    lr_r.fit(X_tr_r, y_tr_aug, lr__sample_weight=w_arr_np)
    p_lr_r = lr_r.predict_proba(X_te_r)
    acc_lr_r, ll_lr_r, br_lr_r = eval_single(p_lr_r, y_te)

    # XGB alone
    xgb_r = XGBClassifier(**XGB_PARAMS)
    xgb_r.fit(X_tr_r, y_tr_aug, sample_weight=w_arr_np)
    p_xgb_r = xgb_r.predict_proba(X_te_r)
    acc_xgb_r, ll_xgb_r, br_xgb_r = eval_single(p_xgb_r, y_te)

    print(f"  LR-{label}:  acc={acc_lr_r*100:.4f}%  ll={ll_lr_r:.6f}")
    print(f"  XGB-{label}: acc={acc_xgb_r*100:.4f}%  ll={ll_xgb_r:.6f}")
    summary_block(f"EXP2 LR-{label}",n_feats,"Elo K=48",acc_lr_r,ll_lr_r,br_lr_r,
                  f"LR only, {n_feats} feats")
    summary_block(f"EXP2 XGB-{label}",n_feats,"Elo K=48",acc_xgb_r,ll_xgb_r,br_xgb_r,
                  f"XGB only, {n_feats} feats")
    exp2_results[label] = {'lr':(acc_lr_r,ll_lr_r,br_lr_r,feat_list,lr_r),
                           'xgb':(acc_xgb_r,ll_xgb_r,br_xgb_r,feat_list,xgb_r)}
    del lr_r, xgb_r; gc.collect()

# Pick best feature set for downstream experiments
best_exp2 = max(
    [(n,m,exp2_results[n][m][0],exp2_results[n][m][3])
     for n in exp2_results for m in exp2_results[n]],
    key=lambda x: x[2]
)
best_n_feats  = best_exp2[0]
best_feats    = best_exp2[3]
best_model_t  = best_exp2[1]
print(f"\n  Best Exp2 subset: {best_n_feats} ({best_model_t}) — acc={best_exp2[2]*100:.4f}%")
print(f"  Using {best_n_feats} features ({len(best_feats)} feats) for Exp 3-5")

X_tr_best = X_tr_aug[best_feats]
X_te_best = X_te[best_feats]
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 3 — LR-Only Baseline (best feature set)")
# ══════════════════════════════════════════════════════════════════════════════

C_VALS = [0.001, 0.01, 0.1, 1.0, 10.0]
skf3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"[{ts()}] CV search over C={C_VALS} on {len(best_feats)} features...")
best_lr_C = None; best_lr_cv = -1
for c in C_VALS:
    pipe = Pipeline([('sc', RobustScaler()),
                     ('lr', LogisticRegression(penalty='l2',C=c,solver='liblinear',
                                               max_iter=2000,random_state=SEED))])
    cv_scores = cross_val_score(pipe, X_tr_best, y_tr_aug, cv=skf3,
                                scoring='accuracy', n_jobs=1, fit_params={'lr__sample_weight': w_arr})
    print(f"  C={c:<6}: CV acc = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    if cv_scores.mean() > best_lr_cv:
        best_lr_cv, best_lr_C = cv_scores.mean(), c

print(f"\n  Best C = {best_lr_C}  (CV acc={best_lr_cv:.4f})")
lr_exp3 = Pipeline([('sc', RobustScaler()),
                    ('lr', LogisticRegression(penalty='l2',C=best_lr_C,solver='liblinear',
                                              max_iter=2000,random_state=SEED))])
lr_exp3.fit(X_tr_best, y_tr_aug, lr__sample_weight=w_arr)
p_exp3 = lr_exp3.predict_proba(X_te_best)
acc_e3, ll_e3, br_e3 = eval_single(p_exp3, y_te)
print(f"  LR-only test: acc={acc_e3*100:.4f}%  ll={ll_e3:.6f}  brier={br_e3:.6f}")
summary_block(f"EXP3: LR-Only (C={best_lr_C}, {len(best_feats)} feats)",len(best_feats),
              "Elo K=48",acc_e3,ll_e3,br_e3,
              f"vs baseline: {(acc_e3-acc_b)*100:+.3f}pp")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 4 — Alternative Ensemble Members")
# ══════════════════════════════════════════════════════════════════════════════

skf4 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
exp4_models = {}

# ── LightGBM default ──────────────────────────────────────────────────────────
print(f"\n[{ts()}] LightGBM default params...")
lgb_def = LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1)
lgb_def.fit(X_tr_best, y_tr_aug, sample_weight=w_arr)
p_lgb_def = lgb_def.predict_proba(X_te_best)
acc_lgb_d, ll_lgb_d, br_lgb_d = eval_single(p_lgb_def, y_te)
print(f"  LGB default: acc={acc_lgb_d*100:.4f}%")
summary_block(f"EXP4: LightGBM Default",len(best_feats),"Elo K=48",
              acc_lgb_d,ll_lgb_d,br_lgb_d,"Default params")
exp4_models['lgb_default'] = (lgb_def, acc_lgb_d)
gc.collect()

# ── LightGBM Optuna ───────────────────────────────────────────────────────────
print(f"[{ts()}] LightGBM Optuna ({N_OPT} trials)...")
def lgb_obj(trial):
    p = {'n_estimators': trial.suggest_int('n_estimators',50,500),
         'learning_rate': trial.suggest_float('learning_rate',0.01,0.3,log=True),
         'max_depth': trial.suggest_int('max_depth',2,8),
         'num_leaves': trial.suggest_int('num_leaves',8,127),
         'subsample': trial.suggest_float('subsample',0.5,1.0),
         'colsample_bytree': trial.suggest_float('colsample_bytree',0.5,1.0),
         'reg_alpha': trial.suggest_float('reg_alpha',0,2),
         'min_child_samples': trial.suggest_int('min_child_samples',5,50)}
    m = LGBMClassifier(**p, random_state=SEED, n_jobs=1, verbose=-1)
    s = cross_val_score(m, X_tr_best, y_tr_aug, cv=skf4, scoring='accuracy',
                        n_jobs=1, fit_params={'sample_weight': w_arr})
    return s.mean()
study_lgb = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lgb_obj, n_trials=N_OPT, show_progress_bar=False)
lgb_opt = LGBMClassifier(**study_lgb.best_params, random_state=SEED, n_jobs=1, verbose=-1)
lgb_opt.fit(X_tr_best, y_tr_aug, sample_weight=w_arr)
p_lgb_opt = lgb_opt.predict_proba(X_te_best)
acc_lgb_o, ll_lgb_o, br_lgb_o = eval_single(p_lgb_opt, y_te)
print(f"  LGB Optuna:  acc={acc_lgb_o*100:.4f}%  (best trial: {study_lgb.best_value:.4f})")
summary_block(f"EXP4: LightGBM Optuna ({N_OPT} trials)",len(best_feats),"Elo K=48",
              acc_lgb_o,ll_lgb_o,br_lgb_o,f"Best CV={study_lgb.best_value:.4f}")
exp4_models['lgb_opt'] = (lgb_opt, acc_lgb_o)
del study_lgb; gc.collect()

# ── CatBoost default ──────────────────────────────────────────────────────────
print(f"[{ts()}] CatBoost default params...")
cat_def = CatBoostClassifier(random_seed=SEED, verbose=False, thread_count=1)
cat_def.fit(X_tr_best, y_tr_aug, sample_weight=w_arr)
p_cat_def = cat_def.predict_proba(X_te_best)
acc_cat_d, ll_cat_d, br_cat_d = eval_single(p_cat_def, y_te)
print(f"  CatBoost default: acc={acc_cat_d*100:.4f}%")
summary_block("EXP4: CatBoost Default",len(best_feats),"Elo K=48",
              acc_cat_d,ll_cat_d,br_cat_d,"Default params")
exp4_models['cat_default'] = (cat_def, acc_cat_d)
gc.collect()

# ── CatBoost Optuna ───────────────────────────────────────────────────────────
print(f"[{ts()}] CatBoost Optuna ({N_OPT} trials)...")
def cat_obj(trial):
    p = {'iterations': trial.suggest_int('iterations',50,500),
         'learning_rate': trial.suggest_float('learning_rate',0.01,0.3,log=True),
         'depth': trial.suggest_int('depth',2,8),
         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg',1,10),
         'subsample': trial.suggest_float('subsample',0.5,1.0)}
    m = CatBoostClassifier(**p, random_seed=SEED, verbose=False, thread_count=1)
    s = cross_val_score(m, X_tr_best, y_tr_aug, cv=skf4, scoring='accuracy', n_jobs=1)
    return s.mean()
study_cat = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
study_cat.optimize(cat_obj, n_trials=N_OPT, show_progress_bar=False)
cat_opt = CatBoostClassifier(**study_cat.best_params, random_seed=SEED, verbose=False, thread_count=1)
cat_opt.fit(X_tr_best, y_tr_aug, sample_weight=w_arr)
p_cat_opt = cat_opt.predict_proba(X_te_best)
acc_cat_o, ll_cat_o, br_cat_o = eval_single(p_cat_opt, y_te)
print(f"  CatBoost Optuna: acc={acc_cat_o*100:.4f}%  (best trial: {study_cat.best_value:.4f})")
summary_block(f"EXP4: CatBoost Optuna ({N_OPT} trials)",len(best_feats),"Elo K=48",
              acc_cat_o,ll_cat_o,br_cat_o,f"Best CV={study_cat.best_value:.4f}")
exp4_models['cat_opt'] = (cat_opt, acc_cat_o)
del study_cat; gc.collect()

# ── Random Forest default ─────────────────────────────────────────────────────
print(f"[{ts()}] Random Forest default params...")
rf_def = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1)
rf_def.fit(X_tr_best, y_tr_aug, sample_weight=w_arr)
p_rf_def = rf_def.predict_proba(X_te_best)
acc_rf_d, ll_rf_d, br_rf_d = eval_single(p_rf_def, y_te)
print(f"  RF default: acc={acc_rf_d*100:.4f}%")
summary_block("EXP4: Random Forest Default",len(best_feats),"Elo K=48",
              acc_rf_d,ll_rf_d,br_rf_d,"200 trees, default depth")
exp4_models['rf_default'] = (rf_def, acc_rf_d)
gc.collect()

# ── Random Forest Optuna ──────────────────────────────────────────────────────
print(f"[{ts()}] Random Forest Optuna ({N_OPT} trials)...")
def rf_obj(trial):
    p = {'n_estimators': trial.suggest_int('n_estimators',50,400),
         'max_depth': trial.suggest_int('max_depth',3,20),
         'min_samples_split': trial.suggest_int('min_samples_split',2,20),
         'min_samples_leaf': trial.suggest_int('min_samples_leaf',1,10),
         'max_features': trial.suggest_float('max_features',0.3,1.0)}
    m = RandomForestClassifier(**p, random_state=SEED, n_jobs=1)
    s = cross_val_score(m, X_tr_best, y_tr_aug, cv=skf4, scoring='accuracy',
                        n_jobs=1, fit_params={'sample_weight': w_arr})
    return s.mean()
study_rf = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_rf.optimize(rf_obj, n_trials=N_OPT, show_progress_bar=False)
rf_opt = RandomForestClassifier(**study_rf.best_params, random_state=SEED, n_jobs=1)
rf_opt.fit(X_tr_best, y_tr_aug, sample_weight=w_arr)
p_rf_opt = rf_opt.predict_proba(X_te_best)
acc_rf_o, ll_rf_o, br_rf_o = eval_single(p_rf_opt, y_te)
print(f"  RF Optuna: acc={acc_rf_o*100:.4f}%  (best trial: {study_rf.best_value:.4f})")
summary_block(f"EXP4: Random Forest Optuna ({N_OPT} trials)",len(best_feats),"Elo K=48",
              acc_rf_o,ll_rf_o,br_rf_o,f"Best CV={study_rf.best_value:.4f}")
exp4_models['rf_opt'] = (rf_opt, acc_rf_o)
del study_rf; gc.collect()

# ── SVM with RBF kernel (sample of 2000 for speed) ────────────────────────────
print(f"[{ts()}] SVM RBF (training on 2000-sample subset for speed)...")
scaler_svm = RobustScaler().fit(X_tr_best)
X_svm_tr_s = scaler_svm.transform(X_tr_best)
X_svm_te   = scaler_svm.transform(X_te_best)
idx_svm    = rng.choice(len(X_svm_tr_s), min(2000, len(X_svm_tr_s)), replace=False)
X_svm_s    = X_svm_tr_s[idx_svm]; y_svm_s = y_tr_aug.values[idx_svm]
w_svm_s    = w_arr[idx_svm]
best_svm   = None; best_svm_cv = -1
for C_svm, gamma_svm in [(0.1,'scale'),(1.0,'scale'),(10.0,'scale'),(1.0,0.01)]:
    svm_m = SVC(kernel='rbf', C=C_svm, gamma=gamma_svm, probability=True, random_state=SEED)
    svm_m.fit(X_svm_s, y_svm_s, sample_weight=w_svm_s)
    cv_svm = cross_val_score(svm_m, X_svm_s, y_svm_s, cv=3, scoring='accuracy', n_jobs=1)
    print(f"  SVM C={C_svm} gamma={gamma_svm}: CV={cv_svm.mean():.4f}")
    if cv_svm.mean() > best_svm_cv:
        best_svm_cv, best_svm = cv_svm.mean(), svm_m
p_svm = best_svm.predict_proba(X_svm_te)
acc_svm, ll_svm, br_svm = eval_single(p_svm, y_te)
print(f"  Best SVM: acc={acc_svm*100:.4f}%")
summary_block("EXP4: SVM RBF (best C/gamma)",len(best_feats),"Elo K=48",
              acc_svm,ll_svm,br_svm,"Trained on 2000-sample subset")
exp4_models['svm'] = (best_svm, acc_svm)
del X_svm_tr_s, X_svm_s; gc.collect()

# Top 2 from Exp4 (for stacking) — excl. SVM (poor scaling)
sorted_exp4 = sorted([(k,v[0],v[1]) for k,v in exp4_models.items() if k != 'svm'],
                     key=lambda x: -x[2])
top2_exp4 = sorted_exp4[:2]
print(f"\n  Top-2 Exp4 models for stacking: {[x[0] for x in top2_exp4]}")

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 5 — Stacking with Meta-Learner")
# ══════════════════════════════════════════════════════════════════════════════

# Base models for stacking: LR (Exp3) + top 2 from Exp4
# Generate OOF predictions on training set
print(f"[{ts()}] Generating OOF predictions for stacking ({CV_FOLDS}-fold)...")
skf5 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

base_names  = ['lr_exp3', top2_exp4[0][0], top2_exp4[1][0]]
base_models = [lr_exp3,   top2_exp4[0][1], top2_exp4[1][1]]

oof_train = np.zeros((len(X_tr_best), len(base_models)))
test_preds = np.zeros((len(X_te_best), len(base_models)))

for mi, (name, model) in enumerate(zip(base_names, base_models)):
    print(f"  OOF for {name}...")
    oof_col = np.zeros(len(X_tr_best))
    te_col  = np.zeros(len(X_te_best))
    for fold, (tr_i, va_i) in enumerate(skf5.split(X_tr_best, y_tr_aug)):
        Xtr_f, Xva_f = X_tr_best.iloc[tr_i], X_tr_best.iloc[va_i]
        ytr_f, wtr_f = y_tr_aug.iloc[tr_i].values, w_arr[tr_i]
        # Clone + fit fresh on fold
        import copy
        m_fold = copy.deepcopy(model)
        try:
            m_fold.fit(Xtr_f, ytr_f, sample_weight=wtr_f)
        except TypeError:
            m_fold.fit(Xtr_f, ytr_f)
        oof_col[va_i]  = m_fold.predict_proba(Xva_f)[:,1]
        te_col        += m_fold.predict_proba(X_te_best)[:,1] / CV_FOLDS
    oof_train[:, mi]  = oof_col
    test_preds[:, mi] = te_col
    gc.collect()

# Meta-learner: logistic regression on OOF predictions
print(f"[{ts()}] Fitting meta-learner LR on OOF predictions...")
meta_lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=SEED)
meta_lr.fit(oof_train, y_tr_aug.values)
p_stack = meta_lr.predict_proba(test_preds)
acc_st, ll_st, br_st = eval_single(p_stack, y_te)
print(f"  Stacked ensemble: acc={acc_st*100:.4f}%  ll={ll_st:.6f}  brier={br_st:.6f}")
print(f"  Meta-learner coefs: {dict(zip(base_names, meta_lr.coef_[0]))}")
summary_block(f"EXP5: Stacking (LR meta + {'+'.join(base_names)})",len(best_feats),
              "Elo K=48",acc_st,ll_st,br_st,
              f"vs baseline: {(acc_st-acc_b)*100:+.3f}pp")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 6 — Best Architecture + Best Glicko-2 Combined")
# ══════════════════════════════════════════════════════════════════════════════

# Identify best non-baseline architecture
all_exp_accs = [
    ('baseline', acc_b, FEAT_129, lr_base, xgb_base, 'lr+xgb'),
    ('exp3_lr',  acc_e3, best_feats, lr_exp3, None, 'lr'),
    ('stacking', acc_st, best_feats, None, None, 'stack'),
    (top2_exp4[0][0], top2_exp4[0][2], best_feats, top2_exp4[0][1], None, 'single'),
]
best_arch = max(all_exp_accs, key=lambda x: x[1])
print(f"\n  Best architecture from Exp 3-5: {best_arch[0]} (acc={best_arch[1]*100:.4f}%)")

# Rebuild Glicko-2 features for the best feature set
print(f"[{ts()}] Rebuilding Glicko-2 merge on experiment-6 feature set ({best_g2_name})...")

# Recompute g2 features on the filtered df
df_g2_6 = merge_g2(df.copy(), g2_hist)
if best_g2_name == "G2-domain":
    g2_s6 = compute_glicko2_domain(df_all, 'KO')
    g2_g6 = compute_glicko2_domain(df_all, 'Sub')
    g2_d6 = compute_glicko2_domain(df_all, 'Decision')
    df_g2_6 = merge_domain_g2(df_g2_6, g2_s6, 'strike')
    df_g2_6 = merge_domain_g2(df_g2_6, g2_g6, 'grapple')
    df_g2_6 = merge_domain_g2(df_g2_6, g2_d6, 'dec')
    del g2_s6, g2_g6, g2_d6

# Replace Elo feats in best_feats with Glicko-2 feats
feat_e6 = [f for f in best_feats if f not in ELO_FEATS] + G2_FEATS
if best_g2_name == "G2-domain":
    feat_e6 = feat_e6 + DOMAIN_FEATS

# Force numeric
for col in G2_ALL_FEATS + (['g2_strike_dif','g2_grapple_dif','g2_dec_dif'] if best_g2_name=="G2-domain" else []):
    df_g2_6[col] = pd.to_numeric(df_g2_6.get(col,0),errors='coerce').fillna(0.0)

X_tr_e6_raw = df_g2_6.loc[train_mask, [f for f in feat_e6 if f in df_g2_6.columns]].reset_index(drop=True)
X_te_e6     = df_g2_6.loc[test_mask,  [f for f in feat_e6 if f in df_g2_6.columns]].reset_index(drop=True)
feat_e6_avail = [f for f in feat_e6 if f in df_g2_6.columns]
X_tr_e6_aug, y_e6_aug, w_e6_aug = corner_flip(X_tr_e6_raw, y_tr_raw, w_raw)

print(f"[{ts()}] Training Exp6 ({best_arch[0]} arch + {best_g2_name})...")
if best_arch[5] == 'lr+xgb':
    lr_e6, xgb_e6, p_lr_e6, p_xgb_e6 = train_baseline_models(
        X_tr_e6_aug, y_e6_aug, w_e6_aug.values, X_te_e6, y_te)
    acc_e6, ll_e6, br_e6, _ = eval_blend(p_lr_e6, p_xgb_e6, y_te)
elif best_arch[5] == 'lr':
    lr_e6 = Pipeline([('sc',RobustScaler()),
                      ('lr',LogisticRegression(penalty='l2',C=best_lr_C,solver='liblinear',
                                               max_iter=2000,random_state=SEED))])
    lr_e6.fit(X_tr_e6_aug, y_e6_aug, lr__sample_weight=w_e6_aug.values)
    p_e6 = lr_e6.predict_proba(X_te_e6)
    acc_e6, ll_e6, br_e6 = eval_single(p_e6, y_te)
else:
    # Best single model from Exp4 + Glicko-2
    m_e6 = sorted_exp4[0][1]
    import copy
    m_e6_new = copy.deepcopy(m_e6)
    try:    m_e6_new.fit(X_tr_e6_aug, y_e6_aug, sample_weight=w_e6_aug.values)
    except: m_e6_new.fit(X_tr_e6_aug, y_e6_aug)
    p_e6 = m_e6_new.predict_proba(X_te_e6)
    acc_e6, ll_e6, br_e6 = eval_single(p_e6, y_te)

print(f"  Exp6 (combined): acc={acc_e6*100:.4f}%  ll={ll_e6:.6f}  brier={br_e6:.6f}")
summary_block(f"EXP6: {best_arch[0]} + {best_g2_name} (combined)",len(feat_e6_avail),
              best_g2_name,acc_e6,ll_e6,br_e6,
              f"vs baseline: {(acc_e6-acc_b)*100:+.3f}pp")
del df_g2_6; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("FINAL SUMMARY — ALL EXPERIMENTS RANKED")
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_SORTED = sorted(RESULTS, key=lambda x: -x['acc'])

print(f"\n  {'Rank':<4}  {'Experiment':<46}  {'Feats':>5}  {'Rating':<20}  {'Acc%':>8}  {'LogLoss':>8}  {'Brier':>8}")
print(f"  {'─'*4}  {'─'*46}  {'─'*5}  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}")
for rank, r in enumerate(RESULTS_SORTED, 1):
    is_best = '◄ BEST' if rank == 1 else ''
    is_base = '◄ BASELINE' if 'BASELINE' in r['name'] else ''
    marker  = is_best or is_base
    print(f"  {rank:<4}  {r['name']:<46}  {r['feat_count']:>5}  {r['rating']:<20}  "
          f"{r['acc']*100:>7.4f}%  {r['log_loss']:>8.6f}  {r['brier']:>8.6f}  {marker}")

best_r   = RESULTS_SORTED[0]
base_r   = next(r for r in RESULTS if 'BASELINE' in r['name'])
delta_pp = (best_r['acc'] - base_r['acc']) * 100

print(f"\n{'─'*70}")
print(f"  Top architecture   : {best_r['name']}")
print(f"  Features           : {best_r['feat_count']}")
print(f"  Rating system      : {best_r['rating']}")
print(f"  Test accuracy      : {best_r['acc']*100:.4f}%")
print(f"  vs baseline ({base_r['acc']*100:.4f}%) : {delta_pp:+.4f}pp")
if delta_pp > 0:
    print(f"  ✓ BEATS BASELINE by {delta_pp:.4f}pp")
else:
    print(f"  ✗ Does NOT beat baseline ({delta_pp:.4f}pp)")
print(f"{'─'*70}")

# Save results JSON
json_path = 'experiments/research/architecture_sprint_results.json'
with open(json_path, 'w') as f:
    json.dump({'baseline_acc': base_r['acc'], 'best_acc': best_r['acc'],
               'delta_pp': delta_pp, 'results': RESULTS_SORTED}, f, indent=2)
print(f"\n  Results saved to {json_path}")
print(f"\n[{ts()}] Sprint complete.")
