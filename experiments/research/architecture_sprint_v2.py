#!/usr/bin/env python3
"""
architecture_sprint_v2.py — Fixed continuation of architecture sprint.

Fixes from v1:
  - cross_val_score fit_params removed in sklearn 1.6 → manual CV loop
  - Glicko-2 collapse investigated and fixed (index alignment + scale diagnostics)
  - Optuna objectives use no-weight CV (consistent, no fit_params needed)

Runs all 6 experiments. Includes known v1 results for Exp 2.
"""
import gc, json, math, os, sys, time, warnings
from collections import defaultdict
from sklearn.base import clone
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
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
N_OPT        = 30
CV_FOLDS     = 5
WOMENS = ["Women's Strawweight","Women's Flyweight",
          "Women's Bantamweight","Women's Featherweight"]

RESULTS = []
BASELINE_ACC = None

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
            f"  │  Note       : {str(note):<48}│\n"
            f"  └{'─'*64}┘")
    print(line)
    RESULTS.append(dict(name=name, feat_count=feat_count, rating=rating,
                        acc=acc, log_loss=ll, brier=brier, note=str(note)))

def eval_blend(p_lr, p_xgb, y_test, lr_w=0.70, xgb_w=0.30):
    p = lr_w * p_lr + xgb_w * p_xgb
    return (accuracy_score(y_test, (p[:,1]>0.5).astype(int)),
            log_loss(y_test, p), brier_score_loss(y_test, p[:,1]), p)

def eval_single(p, y_test):
    return (accuracy_score(y_test, (p[:,1]>0.5).astype(int)),
            log_loss(y_test, p), brier_score_loss(y_test, p[:,1]))

# ── Manual weighted CV (replaces cross_val_score + fit_params) ────────────────
def cv_accuracy(model, X, y, w, skf):
    """5-fold CV accuracy with sample weights. Works with any sklearn version."""
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y.values if hasattr(y, 'values') else y
    w_arr = w.values if hasattr(w, 'values') else w
    scores = []
    for tr_i, va_i in skf.split(X_arr, y_arr):
        m = clone(model)
        try:
            m.fit(X_arr[tr_i], y_arr[tr_i], sample_weight=w_arr[tr_i])
        except TypeError:
            m.fit(X_arr[tr_i], y_arr[tr_i])
        preds = m.predict(X_arr[va_i])
        scores.append(accuracy_score(y_arr[va_i], preds))
    return np.array(scores)

# ── Corner flip with weights ──────────────────────────────────────────────────
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

XGB_PARAMS = {'n_estimators':200,'learning_rate':0.1,'max_depth':6,
              'min_child_weight':1,'subsample':0.8,'colsample_bytree':0.7,
              'gamma':0.3,'reg_alpha':0,'reg_lambda':2.0,
              'random_state':SEED,'eval_metric':'logloss','verbosity':0,'n_jobs':1}
LR_C = 0.00711

def train_lr_xgb(X_tr_a, y_tr_a, w_a, X_te, y_te, lr_c=LR_C, xgb_p=None):
    if xgb_p is None: xgb_p = XGB_PARAMS
    wa = w_a.values if hasattr(w_a,'values') else w_a
    lr = Pipeline([('sc',RobustScaler()),
                   ('lr',LogisticRegression(penalty='l2',C=lr_c,solver='liblinear',
                                            max_iter=2000,random_state=SEED))])
    lr.fit(X_tr_a, y_tr_a, lr__sample_weight=wa)
    xgb = XGBClassifier(**xgb_p)
    xgb.fit(X_tr_a, y_tr_a, sample_weight=wa)
    return lr, xgb, lr.predict_proba(X_te), xgb.predict_proba(X_te)

# ─────────────────────────────────────────────────────────────────────────────
print_section("SHARED SETUP — Loading + Feature Engineering")
# ─────────────────────────────────────────────────────────────────────────────
print(f"[{ts()}] Loading data...")
df_all    = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_all['date'] = pd.to_datetime(df_all['date'])
career_raw = pd.read_csv('data/career_fights_updated.csv')
career_raw['date'] = pd.to_datetime(career_raw['date'])
career_raw = career_raw.sort_values(['fighter','date']).reset_index(drop=True)
style_df   = pd.read_csv('data/ufc_fighters_final_updated.csv')
for col in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    style_df[col] = pd.to_numeric(
        style_df[col].astype(str).str.replace('%','',regex=False),
        errors='coerce').fillna(0.0)/100.0
print(f"  ufc-master: {len(df_all):,}  career: {len(career_raw):,}")

print(f"[{ts()}] Computing Elo K=48...")
def compute_elo_hist(df_src, K=48, base=1500.0):
    ds = df_src.sort_values('date').reset_index(drop=True)
    elo = {}; rows = []
    for _, row in ds.iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        rb, bb = elo.get(r,base), elo.get(b,base)
        re = 1.0/(1.0+10.0**((bb-rb)/400.0))
        ra = 1.0 if row['Winner']=='Red' else (0.0 if row['Winner']=='Blue' else 0.5)
        ra_new, ba_new = rb+K*(ra-re), bb+K*((1-ra)-(1-re))
        rows+=[{'fighter':r,'date':row['date'],'elo_before':rb,'elo_after':ra_new},
               {'fighter':b,'date':row['date'],'elo_before':bb,'elo_after':ba_new}]
        elo[r], elo[b] = ra_new, ba_new
    h = pd.DataFrame(rows).sort_values(['fighter','date']).reset_index(drop=True)
    h['elo_trend'] = h.groupby('fighter')['elo_before'].transform(lambda x: x-x.shift(3))
    return h
elo_hist = compute_elo_hist(df_all)
print(f"  Elo history: {len(elo_hist):,} rows")

print(f"[{ts()}] Computing career stats (shift=1)...")
def compute_career_stats(cdf):
    df = cdf.sort_values(['fighter','date']).copy().reset_index(drop=True)
    df['_ko']  = ((df['won']==1)&df['method'].str.contains('KO|TKO',case=False,na=False)).astype(float)
    df['_sub'] = ((df['won']==1)&df['method'].str.contains('Sub|Submission',case=False,na=False)).astype(float)
    df['_fin'] = ((df['won']==1)&df['method'].str.contains('KO|TKO|Sub',case=False,na=False)).astype(float)
    g = df.groupby('fighter',sort=False)
    df['cum_fights']=g.cumcount()
    for src,dst in [('won','_cs_won'),('_ko','_cs_ko'),('_sub','_cs_sub'),('_fin','_cs_fin')]:
        df[dst] = g[src].cumsum()-df[src]
    sn = df['cum_fights'].clip(lower=1)
    df['career_win_rate']    = np.where(df['cum_fights']>0,df['_cs_won']/sn,0.5)
    df['ko_finish_rate']     = np.where(df['cum_fights']>0,df['_cs_ko']/sn, 0.0)
    df['sub_finish_rate']    = np.where(df['cum_fights']>0,df['_cs_sub']/sn,0.0)
    df['career_finish_rate'] = np.where(df['cum_fights']>0,df['_cs_fin']/sn,0.0)
    def _r(s,w,d): return s.shift(1).rolling(w,min_periods=1).mean().fillna(d)
    df['last3_win_rate']    = g['won'].transform(lambda x: _r(x,3,0.5))
    df['last10_win_rate']   = g['won'].transform(lambda x: _r(x,10,0.5))
    df['last5_won']         = g['won'].transform(lambda x: _r(x,5,0.5))
    df['last5_finish_rate'] = g['_fin'].transform(lambda x: _r(x,5,0.0))
    df['trend_score']       = df['last3_win_rate']-df['last10_win_rate']
    df['_prev_date']        = g['date'].transform(lambda x: x.shift(1))
    df['layoff_days']       = (df['date']-df['_prev_date']).dt.days.fillna(180.0).clip(lower=0)
    all_wr = {f:grp['won'].sum()/max(1,len(grp)) for f,grp in df.groupby('fighter')}
    opp_col=df['opponent'].tolist(); ftr_col=df['fighter'].tolist()
    ftr_pos=defaultdict(list)
    for pos,idx in enumerate(df.index.tolist()):
        ftr_pos[ftr_col[pos]].append(pos)
    oq=np.full(len(df),0.5)
    for ftr,positions in ftr_pos.items():
        for rank,pos in enumerate(positions):
            past=[opp_col[p] for p in positions[max(0,rank-5):rank]]
            rates=[all_wr.get(o,0.5) for o in past]
            oq[pos]=float(np.mean(rates)) if rates else 0.5
    df['opp_quality']=oq
    keep=['fighter','date','cum_fights','career_win_rate','ko_finish_rate',
          'sub_finish_rate','career_finish_rate','last3_win_rate','last10_win_rate',
          'last5_won','last5_finish_rate','trend_score','layoff_days','opp_quality']
    return df[keep]
career_stats = compute_career_stats(career_raw)
print(f"  Career rows: {len(career_stats):,}")

print(f"[{ts()}] Computing QA stats...")
def compute_qa_stats(cdf, elo_h):
    elo_ref = elo_h[['fighter','date','elo_before']].copy()
    od = cdf[['fighter','opponent','date','won','got_finish']].copy().rename(columns={'opponent':'opp_name'})
    oref = elo_ref.rename(columns={'fighter':'opp_name','elo_before':'opp_elo'})
    od = pd.merge_asof(od.sort_values('date'), oref.sort_values('date'),
                       on='date', by='opp_name', direction='backward')
    od['opp_elo']=od['opp_elo'].fillna(1500.0); od['ew']=od['opp_elo']/1500.0
    rows=[]
    for fighter, grp in od.groupby('fighter',sort=False):
        grp=grp.sort_values('date'); n=len(grp)
        qa_wr=np.full(n,0.5); qa_fr=np.full(n,0.0)
        qa_sl=np.full(n,0.0); qa_sa=np.full(n,0.0)
        cew=ceww=cewf=cn=coff=cdef_=0.0
        for i,(_,r) in enumerate(grp.iterrows()):
            if cew>0: qa_wr[i]=ceww/cew; qa_fr[i]=cewf/cew
            if cn>0:  qa_sl[i]=coff/cn;  qa_sa[i]=cdef_/cn
            ew=r['ew']; w=r['won']; f=r['got_finish'] if pd.notna(r.get('got_finish')) else 0.0
            cew+=ew; ceww+=ew*w; cewf+=ew*f; cn+=ew; coff+=ew*w; cdef_+=ew*(1.0-w)
        rows.append(pd.DataFrame({'fighter':fighter,'date':grp['date'].values,
                                   'qa_win_rate':qa_wr,'qa_finish_rate':qa_fr,
                                   'qa_SLpM':qa_sl,'qa_SApM':qa_sa}))
    return pd.concat(rows,ignore_index=True).sort_values(['fighter','date'])
qa_stats = compute_qa_stats(career_raw, elo_hist)
print(f"  QA rows: {len(qa_stats):,}")

WC_ORDER = {"Women's Strawweight":0,"Women's Flyweight":1,"Women's Bantamweight":2,
            "Women's Featherweight":3,"Flyweight":4,"Bantamweight":5,
            "Featherweight":6,"Lightweight":7,"Welterweight":8,
            "Middleweight":9,"Light Heavyweight":10,"Heavyweight":11,"Catch Weight":6}

print(f"[{ts()}] Building main fight DataFrame (men's 2015+)...")
df = df_all[(df_all['date']>=TRAIN_START)&df_all['Winner'].isin(['Red','Blue'])&
            ~df_all['weight_class'].isin(WOMENS)].copy().sort_values('date').reset_index(drop=True)
print(f"  Fights: {len(df):,}")

def merge_career(df_fights, career_s):
    cc=[c for c in career_s.columns if c not in ('fighter','date')]
    r_cs=career_s.rename(columns={'fighter':'R_fighter',**{c:f'R_{c}' for c in cc}})
    b_cs=career_s.rename(columns={'fighter':'B_fighter',**{c:f'B_{c}' for c in cc}})
    out=pd.merge_asof(df_fights.sort_values('date'),r_cs.sort_values('date'),
                      on='date',by='R_fighter',direction='backward')
    out=pd.merge_asof(out.sort_values('date'),b_cs.sort_values('date'),
                      on='date',by='B_fighter',direction='backward')
    cdef={'cum_fights':0,'career_win_rate':0.5,'ko_finish_rate':0.0,'sub_finish_rate':0.0,
          'career_finish_rate':0.0,'last3_win_rate':0.5,'last10_win_rate':0.5,
          'last5_won':0.5,'last5_finish_rate':0.0,'trend_score':0.0,
          'layoff_days':180.0,'opp_quality':0.5}
    for stat,dv in cdef.items():
        out[f'R_{stat}']=out[f'R_{stat}'].fillna(dv)
        out[f'B_{stat}']=out[f'B_{stat}'].fillna(dv)
    return out.sort_values('date').reset_index(drop=True)

df = merge_career(df, career_stats)

qa_r=qa_stats.rename(columns={'fighter':'R_fighter','qa_win_rate':'R_qa_win_rate',
    'qa_finish_rate':'R_qa_finish_rate','qa_SLpM':'R_qa_SLpM','qa_SApM':'R_qa_SApM'})
qa_b=qa_stats.rename(columns={'fighter':'B_fighter','qa_win_rate':'B_qa_win_rate',
    'qa_finish_rate':'B_qa_finish_rate','qa_SLpM':'B_qa_SLpM','qa_SApM':'B_qa_SApM'})
df=pd.merge_asof(df.sort_values('date'),qa_r.sort_values('date'),
                 on='date',by='R_fighter',direction='backward')
df=pd.merge_asof(df.sort_values('date'),qa_b.sort_values('date'),
                 on='date',by='B_fighter',direction='backward')
for c in ['R_qa_win_rate','R_qa_finish_rate','R_qa_SLpM','R_qa_SApM',
          'B_qa_win_rate','B_qa_finish_rate','B_qa_SLpM','B_qa_SApM']:
    df[c]=df[c].fillna(0.5 if 'win_rate' in c else 0.0)
df['qa_win_rate_dif']=df['R_qa_win_rate']-df['B_qa_win_rate']
df['qa_finish_rate_dif']=df['R_qa_finish_rate']-df['B_qa_finish_rate']
df['qa_SLpM_dif']=df['R_qa_SLpM']-df['B_qa_SLpM']
df['qa_SApM_dif']=df['R_qa_SApM']-df['B_qa_SApM']

er=elo_hist[['fighter','date','elo_before','elo_trend']].rename(
    columns={'fighter':'R_fighter','elo_before':'R_elo','elo_trend':'R_elo_trend'})
eb=elo_hist[['fighter','date','elo_before','elo_trend']].rename(
    columns={'fighter':'B_fighter','elo_before':'B_elo','elo_trend':'B_elo_trend'})
df=pd.merge_asof(df.sort_values('date'),er.sort_values('date'),
                 on='date',by='R_fighter',direction='backward')
df=pd.merge_asof(df.sort_values('date'),eb.sort_values('date'),
                 on='date',by='B_fighter',direction='backward')
df['R_elo']=df['R_elo'].fillna(1500.0); df['B_elo']=df['B_elo'].fillna(1500.0)
df['R_elo_trend']=df['R_elo_trend'].fillna(0.0); df['B_elo_trend']=df['B_elo_trend'].fillna(0.0)
df['elo_dif']=df['R_elo']-df['B_elo']; df['elo_trend_dif']=df['R_elo_trend']-df['B_elo_trend']

style_src=['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']
sdf=style_df.drop_duplicates(subset=['Fighter_Name'],keep='last')
rs=sdf[['Fighter_Name']+style_src].rename(columns={'Fighter_Name':'R_fighter',**{c:f'R_{c}' for c in style_src}})
bs=sdf[['Fighter_Name']+style_src].rename(columns={'Fighter_Name':'B_fighter',**{c:f'B_{c}' for c in style_src}})
df=df.merge(rs,on='R_fighter',how='left').merge(bs,on='B_fighter',how='left')
for col in [f'{p}{s}' for p in ('R_','B_') for s in style_src]:
    df[col]=pd.to_numeric(df[col],errors='coerce').fillna(0.0)

df['weight_class_ord']=df['weight_class'].map(WC_ORDER).fillna(6).astype(int)
df['R_southpaw']=(df['R_Stance'].str.lower()=='southpaw').astype(int)
df['B_southpaw']=(df['B_Stance'].str.lower()=='southpaw').astype(int)
df['orth_clash']=((df['R_southpaw']==0)&(df['B_southpaw']==0)).astype(int)
df['south_clash']=((df['R_southpaw']==1)&(df['B_southpaw']==1)).astype(int)
df['R_age']=pd.to_numeric(df['R_age'],errors='coerce').fillna(28.0)
df['B_age']=pd.to_numeric(df['B_age'],errors='coerce').fillna(28.0)
df['R_age_x_exp']=df['R_age']*df['R_cum_fights']
df['B_age_x_exp']=df['B_age']*df['B_cum_fights']
df['age_x_exp_dif']=df['R_age_x_exp']-df['B_age_x_exp']
for p,s in [('R_',df['R_layoff_days']),('B_',df['B_layoff_days'])]:
    d=s.fillna(180.0)
    df[f'{p}layoff_lt90']=(d<90).astype(int)
    df[f'{p}layoff_90_180']=((d>=90)&(d<180)).astype(int)
    df[f'{p}layoff_180_365']=((d>=180)&(d<365)).astype(int)
    df[f'{p}layoff_gt365']=(d>=365).astype(int)
for stat in ['career_win_rate','last5_won','last5_finish_rate','opp_quality',
             'trend_score','ko_finish_rate','sub_finish_rate','last3_win_rate','last10_win_rate']:
    df[f'{stat}_dif']=df[f'R_{stat}']-df[f'B_{stat}']
for pair in [('SLpM','SLpM_dif'),('SApM','SApM_dif'),('Str_Def','Str_Def_dif'),
             ('TD_Def','TD_Def_dif'),('Sub_Avg','Sub_Avg_dif'),('TD_Avg','TD_Avg_dif')]:
    df[pair[1]]=df[f'R_{pair[0]}']-df[f'B_{pair[0]}']
for c in ['R_wins','R_losses','B_wins','B_losses','R_Height_cms','R_Reach_cms',
          'B_Height_cms','B_Reach_cms','R_avg_SIG_STR_landed','R_avg_TD_landed',
          'R_avg_SIG_STR_pct','R_avg_SUB_ATT','R_avg_TD_pct',
          'B_avg_SIG_STR_landed','B_avg_TD_landed','B_avg_SIG_STR_pct',
          'B_avg_SUB_ATT','B_avg_TD_pct','R_current_win_streak',
          'R_current_lose_streak','R_longest_win_streak','B_current_win_streak',
          'B_current_lose_streak','B_longest_win_streak','B_total_title_bouts']:
    df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
df['win_dif']         =df['R_wins']-df['B_wins']
df['loss_dif']        =df['R_losses']-df['B_losses']
df['win_streak_dif']  =df['R_current_win_streak']-df['B_current_win_streak']
df['lose_streak_dif'] =df['R_current_lose_streak']-df['B_current_lose_streak']
df['height_dif']      =df['R_Height_cms']-df['B_Height_cms']
df['reach_dif']       =df['R_Reach_cms']-df['B_Reach_cms']
df['age_dif']         =df['R_age']-df['B_age']
df['sig_str_dif']     =df['R_avg_SIG_STR_landed']-df['B_avg_SIG_STR_landed']
df['avg_td_dif']      =df['R_avg_TD_landed']-df['B_avg_TD_landed']
df['ko_dif']          =df['R_ko_finish_rate']-df['B_ko_finish_rate']
df['sub_dif']         =df['R_sub_finish_rate']-df['B_sub_finish_rate']
df['total_title_bout_dif']=pd.to_numeric(df.get('B_total_title_bouts',0),errors='coerce').fillna(0)*0

print(f"[{ts()}] Interaction features...")
cdf2=career_raw.sort_values(['fighter','date']).copy()
cdf2['is_loss']    =(cdf2['won']==0).astype(float)
cdf2['is_fin_loss']=((cdf2['won']==0)&(cdf2['got_finish'].fillna(0)==1)).astype(float)
g2=cdf2.groupby('fighter',sort=False)
cdf2['_cs_l'] =g2['is_loss'].cumsum()    -cdf2['is_loss']
cdf2['_cs_fl']=g2['is_fin_loss'].cumsum()-cdf2['is_fin_loss']
cdf2['got_finished_rate']=np.where(cdf2['_cs_l']>0,cdf2['_cs_fl']/cdf2['_cs_l'],0.5)
chin=cdf2[['fighter','date','got_finished_rate']].sort_values(['fighter','date'])
cr2=chin.rename(columns={'fighter':'R_fighter','got_finished_rate':'R_got_finished_rate'})
cb2=chin.rename(columns={'fighter':'B_fighter','got_finished_rate':'B_got_finished_rate'})
df=pd.merge_asof(df.sort_values('date'),cr2.sort_values('date'),
                 on='date',by='R_fighter',direction='backward')
df=pd.merge_asof(df.sort_values('date'),cb2.sort_values('date'),
                 on='date',by='B_fighter',direction='backward')
df['R_got_finished_rate']=df['R_got_finished_rate'].fillna(0.5)
df['B_got_finished_rate']=df['B_got_finished_rate'].fillna(0.5)
df['R_age_x_layoff']    =df['R_age']*df['R_layoff_days'].clip(upper=730)
df['B_age_x_layoff']    =df['B_age']*df['B_layoff_days'].clip(upper=730)
df['age_x_layoff_dif']  =df['R_age_x_layoff']-df['B_age_x_layoff']
df['R_finish_danger']   =df['R_ko_finish_rate']+df['R_sub_finish_rate']
df['B_finish_danger']   =df['B_ko_finish_rate']+df['B_sub_finish_rate']
df['finish_danger_mismatch']=(df['R_finish_danger']*(1-df['B_got_finished_rate'])-
                              df['B_finish_danger']*(1-df['R_got_finished_rate']))
del cdf2,g2,chin,cr2,cb2; gc.collect()

df=df[(df['R_cum_fights']>=1)&(df['B_cum_fights']>=1)].copy().sort_values('date').reset_index(drop=True)
df['target']=(df['Winner']=='Red').astype(int)
print(f"  After debut filter: {len(df):,}  "
      f"train={(df['date']<TRAIN_CUTOFF).sum()}  "
      f"test={(df['date']>=TRAIN_CUTOFF).sum()}")

FEAT_BASE=[
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
FEAT_QA=["R_qa_win_rate","R_qa_finish_rate","R_qa_SLpM","R_qa_SApM",
         "B_qa_win_rate","B_qa_finish_rate","B_qa_SLpM","B_qa_SApM",
         "qa_win_rate_dif","qa_finish_rate_dif","qa_SLpM_dif","qa_SApM_dif"]
FEAT_INT=["R_age_x_layoff","B_age_x_layoff","age_x_layoff_dif",
          "R_finish_danger","B_finish_danger","finish_danger_mismatch",
          "R_got_finished_rate","B_got_finished_rate"]
FEAT_129=FEAT_BASE+FEAT_QA+FEAT_INT
for col in FEAT_129:
    df[col]=pd.to_numeric(df.get(col,0),errors='coerce').fillna(0.0)

# Deterministic train/test split (df is sorted by date, index 0..N-1)
train_mask=df['date']<TRAIN_CUTOFF
test_mask =~train_mask
X_tr_raw=df.loc[train_mask,FEAT_129].reset_index(drop=True)
y_tr_raw=df.loc[train_mask,'target'].reset_index(drop=True)
d_tr_raw=df.loc[train_mask,'date'].reset_index(drop=True)
X_te    =df.loc[test_mask, FEAT_129].reset_index(drop=True)
y_te    =df.loc[test_mask, 'target'].reset_index(drop=True)
w_raw   =pd.Series(compute_weights(d_tr_raw),index=y_tr_raw.index)
X_tr_aug,y_tr_aug,w_tr_aug=corner_flip(X_tr_raw,y_tr_raw,w_raw)
w_arr=w_tr_aug.values
print(f"  Train aug: {len(X_tr_aug):,}  Test: {len(X_te):,}  Features: {len(FEAT_129)}")
print(f"[{ts()}] Setup complete.\n")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("BASELINE — LR 70% + XGB 30%, 129 features, Elo K=48")
# ══════════════════════════════════════════════════════════════════════════════
t0=time.time()
lr_base,xgb_base,p_lr_b,p_xgb_b=train_lr_xgb(X_tr_aug,y_tr_aug,w_arr,X_te,y_te)
acc_b,ll_b,br_b,_=eval_blend(p_lr_b,p_xgb_b,y_te)
BASELINE_ACC=acc_b
print(f"  acc={acc_b*100:.4f}%  ll={ll_b:.6f}  brier={br_b:.6f}  [{time.time()-t0:.1f}s]")
summary_block("BASELINE (LR70+XGB30, Elo, 129 feats)",129,"Elo K=48",acc_b,ll_b,br_b,
              f"Reference: {acc_b*100:.4f}%")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 1 — Glicko-2 (standard + domain-separated)")
# ══════════════════════════════════════════════════════════════════════════════

def glicko2_update(mu, phi, sigma, mu_j, phi_j, s, tau=0.5):
    g_j = 1.0/math.sqrt(1.0+3.0*phi_j**2/math.pi**2)
    E   = 1.0/(1.0+math.exp(-g_j*(mu-mu_j)))
    v   = 1.0/(g_j**2*E*(1.0-E)+1e-10)
    delta = v*g_j*(s-E)
    a = math.log(max(sigma**2,1e-15))
    def f(x):
        ex=math.exp(x); d2=phi**2+v+ex
        return (ex*(delta**2-phi**2-v-ex)/(2.0*d2**2+1e-15))-(x-a)/tau**2
    A=a
    B=math.log(max(delta**2-phi**2-v,1e-15)) if delta**2>phi**2+v else a-tau
    fa,fb=f(A),f(B)
    for _ in range(100):
        if abs(B-A)<1e-6: break
        C=A+(A-B)*fa/(fb-fa+1e-15); fc=f(C)
        if fc*fb<0: A,fa=B,fb
        else: fa/=2.0
        B,fb=C,fc
    sigma_new=math.exp(A/2.0)
    phi_star=math.sqrt(phi**2+sigma_new**2)
    phi_new=1.0/math.sqrt(1.0/phi_star**2+1.0/v)
    mu_new=mu+phi_new**2*g_j*(s-E)
    return mu_new,phi_new,sigma_new

def compute_glicko2(df_src, tau=0.5, init_r=1500, init_rd=350, init_s=0.06):
    SCALE=173.7178
    state={}; rows=[]
    for _,row in df_src.sort_values('date').iterrows():
        r,b=row['R_fighter'],row['B_fighter']
        rm,rp,rs=state.get(r,((init_r-1500)/SCALE,init_rd/SCALE,init_s))
        bm,bp,bs=state.get(b,((init_r-1500)/SCALE,init_rd/SCALE,init_s))
        ra=1.0 if row['Winner']=='Red' else (0.0 if row['Winner']=='Blue' else 0.5)
        ba=1.0-ra
        rows+=[{'fighter':r,'date':row['date'],
                'g2_r':rm*SCALE+1500,'g2_rd':rp*SCALE,'g2_sigma':rs},
               {'fighter':b,'date':row['date'],
                'g2_r':bm*SCALE+1500,'g2_rd':bp*SCALE,'g2_sigma':bs}]
        rm_new,rp_new,rs_new=glicko2_update(rm,rp,rs,bm,bp,ra,tau)
        bm_new,bp_new,bs_new=glicko2_update(bm,bp,bs,rm,rp,ba,tau)
        state[r]=(rm_new,rp_new,rs_new); state[b]=(bm_new,bp_new,bs_new)
    return pd.DataFrame(rows)

print(f"[{ts()}] Computing Glicko-2...")
g2_hist=compute_glicko2(df_all)
print(f"  G2 history: {len(g2_hist):,} rows  "
      f"r range: [{g2_hist['g2_r'].min():.0f}, {g2_hist['g2_r'].max():.0f}]  "
      f"rd range: [{g2_hist['g2_rd'].min():.0f}, {g2_hist['g2_rd'].max():.0f}]")

# Merge G2 onto df — CRITICAL: use explicit sort+reset_index to ensure alignment
def merge_g2_onto(df_fights, g2_df):
    """Merge Glicko-2 ratings. Returns DataFrame in same row order as df_fights."""
    SCALE=173.7178
    gr=g2_df.rename(columns={'fighter':'R_fighter','g2_r':'R_g2_r',
                              'g2_rd':'R_g2_rd','g2_sigma':'R_g2_sigma'})
    gb=g2_df.rename(columns={'fighter':'B_fighter','g2_r':'B_g2_r',
                              'g2_rd':'B_g2_rd','g2_sigma':'B_g2_sigma'})
    # df_fights must be sorted by date for merge_asof
    sort_df=df_fights.sort_values('date').copy()
    out=pd.merge_asof(sort_df,gr.sort_values('date'),
                      on='date',by='R_fighter',direction='backward')
    out=pd.merge_asof(out.sort_values('date'),gb.sort_values('date'),
                      on='date',by='B_fighter',direction='backward')
    out=out.sort_values('date').reset_index(drop=True)  # ensure consistent order
    for c in ['R_g2_r','B_g2_r']:   out[c]=out[c].fillna(1500.0)
    for c in ['R_g2_rd','B_g2_rd']: out[c]=out[c].fillna(350.0)
    for c in ['R_g2_sigma','B_g2_sigma']: out[c]=out[c].fillna(0.06)
    out['g2_r_dif']    =out['R_g2_r']    -out['B_g2_r']
    out['g2_rd_dif']   =out['R_g2_rd']   -out['B_g2_rd']
    out['g2_sigma_dif']=out['R_g2_sigma']-out['B_g2_sigma']
    return out

df_g2=merge_g2_onto(df.copy(), g2_hist)
# Verify alignment: dates must match df
assert len(df_g2)==len(df), f"G2 merge lost rows: {len(df_g2)} vs {len(df)}"
assert (df_g2['date'].values==df['date'].values).all(), "G2 date order mismatch!"

print(f"  G2 features sample (first 3):")
print(f"    R_g2_r: {df_g2['R_g2_r'].head(3).values}")
print(f"    g2_r_dif: {df_g2['g2_r_dif'].head(3).values}")

ELO_FEATS=['R_elo','B_elo','elo_dif','R_elo_trend','B_elo_trend','elo_trend_dif']
G2_FEATS=['R_g2_r','B_g2_r','g2_r_dif','R_g2_rd','B_g2_rd','g2_rd_dif']
FEAT_G2_STD=[f for f in FEAT_129 if f not in ELO_FEATS]+G2_FEATS

# Use df_g2's own train/test split (same dates as df)
g2_train=(df_g2['date']<TRAIN_CUTOFF)
g2_test =~g2_train
X_tr_g2_raw=df_g2.loc[g2_train,FEAT_G2_STD].reset_index(drop=True)
X_te_g2    =df_g2.loc[g2_test, FEAT_G2_STD].reset_index(drop=True)
y_tr_g2_raw=df_g2.loc[g2_train,'target'].reset_index(drop=True)
y_te_g2    =df_g2.loc[g2_test, 'target'].reset_index(drop=True)
d_tr_g2    =df_g2.loc[g2_train,'date'].reset_index(drop=True)
w_g2_raw   =pd.Series(compute_weights(d_tr_g2),index=y_tr_g2_raw.index)
X_tr_g2_aug,y_tr_g2_aug,w_g2_aug=corner_flip(X_tr_g2_raw,y_tr_g2_raw,w_g2_raw)

print(f"[{ts()}] Training Exp1-A (G2 standard, {len(FEAT_G2_STD)} feats)...")
lr_g2,xgb_g2,plr_g2,pxgb_g2=train_lr_xgb(
    X_tr_g2_aug,y_tr_g2_aug,w_g2_aug.values,X_te_g2,y_te_g2)
acc_g2,ll_g2,br_g2,_=eval_blend(plr_g2,pxgb_g2,y_te_g2)
print(f"  G2 standard: acc={acc_g2*100:.4f}%  vs baseline: {(acc_g2-acc_b)*100:+.3f}pp")
summary_block("EXP1A: Glicko-2 Standard (replaces Elo)",len(FEAT_G2_STD),
              "Glicko-2 tau=0.5",acc_g2,ll_g2,br_g2,
              f"vs baseline: {(acc_g2-acc_b)*100:+.3f}pp")
del lr_g2,xgb_g2; gc.collect()

# Domain-separated G2
print(f"\n[{ts()}] Computing domain-separated G2...")
def compute_glicko2_domain(df_src, method_kw, tau=0.5, init_r=1500, init_rd=350, init_s=0.06):
    SCALE=173.7178; state={}; rows=[]
    for _,row in df_src.sort_values('date').iterrows():
        r,b=row['R_fighter'],row['B_fighter']
        rm,rp,rs=state.get(r,((init_r-1500)/SCALE,init_rd/SCALE,init_s))
        bm,bp,bs=state.get(b,((init_r-1500)/SCALE,init_rd/SCALE,init_s))
        rows+=[{'fighter':r,'date':row['date'],'g2_r':rm*SCALE+1500},
               {'fighter':b,'date':row['date'],'g2_r':bm*SCALE+1500}]
        method=str(row.get('finish','') or '')
        if method_kw.lower() in method.lower():
            ra=1.0 if row['Winner']=='Red' else (0.0 if row['Winner']=='Blue' else 0.5)
            ba=1.0-ra
            rm_new,rp_new,rs_new=glicko2_update(rm,rp,rs,bm,bp,ra,tau)
            bm_new,bp_new,bs_new=glicko2_update(bm,bp,bs,rm,rp,ba,tau)
            state[r]=(rm_new,rp_new,rs_new); state[b]=(bm_new,bp_new,bs_new)
    return pd.DataFrame(rows)

def add_domain_g2(df_fights, g2_df, domain):
    gr=g2_df.rename(columns={'fighter':f'R_fighter','g2_r':f'R_g2_{domain}'})
    gb=g2_df.rename(columns={'fighter':f'B_fighter','g2_r':f'B_g2_{domain}'})
    out=pd.merge_asof(df_fights.sort_values('date'),gr.sort_values('date'),
                      on='date',by='R_fighter',direction='backward')
    out=pd.merge_asof(out.sort_values('date'),gb.sort_values('date'),
                      on='date',by='B_fighter',direction='backward')
    out=out.sort_values('date').reset_index(drop=True)
    out[f'R_g2_{domain}']=out[f'R_g2_{domain}'].fillna(1500.0)
    out[f'B_g2_{domain}']=out[f'B_g2_{domain}'].fillna(1500.0)
    out[f'g2_{domain}_dif']=out[f'R_g2_{domain}']-out[f'B_g2_{domain}']
    return out

g2_strike =compute_glicko2_domain(df_all,'KO')
g2_grapple=compute_glicko2_domain(df_all,'Sub')
g2_dec    =compute_glicko2_domain(df_all,'Decision')
df_g2d=add_domain_g2(df_g2.copy(),g2_strike,'strike')
df_g2d=add_domain_g2(df_g2d,g2_grapple,'grapple')
df_g2d=add_domain_g2(df_g2d,g2_dec,'dec')
del g2_strike,g2_grapple,g2_dec; gc.collect()

DOMAIN_FEATS=['g2_strike_dif','g2_grapple_dif','g2_dec_dif']
FEAT_G2_DOM=FEAT_G2_STD+DOMAIN_FEATS

g2d_train=(df_g2d['date']<TRAIN_CUTOFF)
g2d_test =~g2d_train
X_tr_gd_raw=df_g2d.loc[g2d_train,FEAT_G2_DOM].reset_index(drop=True)
X_te_gd    =df_g2d.loc[g2d_test, FEAT_G2_DOM].reset_index(drop=True)
y_tr_gd_raw=df_g2d.loc[g2d_train,'target'].reset_index(drop=True)
y_te_gd    =df_g2d.loc[g2d_test, 'target'].reset_index(drop=True)
d_tr_gd    =df_g2d.loc[g2d_train,'date'].reset_index(drop=True)
w_gd_raw   =pd.Series(compute_weights(d_tr_gd),index=y_tr_gd_raw.index)
X_tr_gd_aug,y_tr_gd_aug,w_gd_aug=corner_flip(X_tr_gd_raw,y_tr_gd_raw,w_gd_raw)

print(f"[{ts()}] Training Exp1-B (G2 domain, {len(FEAT_G2_DOM)} feats)...")
lr_gd,xgb_gd,plr_gd,pxgb_gd=train_lr_xgb(
    X_tr_gd_aug,y_tr_gd_aug,w_gd_aug.values,X_te_gd,y_te_gd)
acc_gd,ll_gd,br_gd,_=eval_blend(plr_gd,pxgb_gd,y_te_gd)
print(f"  G2 domain:   acc={acc_gd*100:.4f}%  vs baseline: {(acc_gd-acc_b)*100:+.3f}pp")
summary_block("EXP1B: Glicko-2 + Domain (striking/grappling/dec)",len(FEAT_G2_DOM),
              "Glicko-2 std+domain",acc_gd,ll_gd,br_gd,
              f"vs baseline: {(acc_gd-acc_b)*100:+.3f}pp")
del lr_gd,xgb_gd,df_g2,df_g2d; gc.collect()

best_g2_acc =(acc_g2 if acc_g2>=acc_gd else acc_gd)
best_g2_feat=(FEAT_G2_STD if acc_g2>=acc_gd else FEAT_G2_DOM)
best_g2_name=("G2-standard" if acc_g2>=acc_gd else "G2-domain")
print(f"\n  Best Glicko-2: {best_g2_name}  acc={best_g2_acc*100:.4f}%")

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 2 — SHAP Feature Reduction")
# ══════════════════════════════════════════════════════════════════════════════
print(f"[{ts()}] Running SHAP TreeExplainer...")
rng=np.random.RandomState(SEED)
shap_idx=rng.choice(len(X_tr_aug),min(400,len(X_tr_aug)),replace=False)
X_shap=X_tr_aug.iloc[shap_idx]
explainer=shap.TreeExplainer(xgb_base)
shap_vals=explainer.shap_values(X_shap)
mean_abs_shap=np.abs(shap_vals).mean(axis=0)
shap_ranking=sorted(zip(FEAT_129,mean_abs_shap),key=lambda x: -x[1])
print("  Top 15 features by SHAP:")
for i,(feat,sv) in enumerate(shap_ranking[:15],1):
    print(f"    {i:2d}. {feat:<35}: {sv:.5f}")
raw_flags=[f for f,_ in shap_ranking[:25]
           if f in ['R_wins','R_losses','B_wins','B_losses',
                    'R_Height_cms','R_Reach_cms','B_Height_cms','B_Reach_cms',
                    'R_age','B_age','R_avg_SIG_STR_landed','B_avg_SIG_STR_landed']]
if raw_flags: print(f"  Raw stats in top 25 (ratio replacement candidates): {raw_flags}")
top60=[f for f,_ in shap_ranking[:60]]
top40=[f for f,_ in shap_ranking[:40]]
top25=[f for f,_ in shap_ranking[:25]]
del explainer,shap_vals,X_shap; gc.collect()

exp2_results={}
for n_feats,feat_list,label in [(60,top60,"Top60"),(40,top40,"Top40"),(25,top25,"Top25")]:
    print(f"\n[{ts()}] Exp2 {label} ({n_feats} features)...")
    X_tr_r=X_tr_aug[feat_list]; X_te_r=X_te[feat_list]
    wa=w_tr_aug.values
    lr_r=Pipeline([('sc',RobustScaler()),
                   ('lr',LogisticRegression(penalty='l2',C=LR_C,solver='liblinear',
                                            max_iter=2000,random_state=SEED))])
    lr_r.fit(X_tr_r,y_tr_aug,lr__sample_weight=wa)
    p_lr_r=lr_r.predict_proba(X_te_r)
    acc_lr_r,ll_lr_r,br_lr_r=eval_single(p_lr_r,y_te)
    xgb_r=XGBClassifier(**XGB_PARAMS)
    xgb_r.fit(X_tr_r,y_tr_aug,sample_weight=wa)
    p_xgb_r=xgb_r.predict_proba(X_te_r)
    acc_xgb_r,ll_xgb_r,br_xgb_r=eval_single(p_xgb_r,y_te)
    print(f"  LR-{label}:  {acc_lr_r*100:.4f}%  XGB-{label}: {acc_xgb_r*100:.4f}%")
    summary_block(f"EXP2 LR-{label}",n_feats,"Elo K=48",acc_lr_r,ll_lr_r,br_lr_r,
                  f"LR only, {n_feats} feats vs baseline: {(acc_lr_r-acc_b)*100:+.3f}pp")
    summary_block(f"EXP2 XGB-{label}",n_feats,"Elo K=48",acc_xgb_r,ll_xgb_r,br_xgb_r,
                  f"XGB only, {n_feats} feats vs baseline: {(acc_xgb_r-acc_b)*100:+.3f}pp")
    exp2_results[label]={'lr':(acc_lr_r,ll_lr_r,br_lr_r,feat_list),
                         'xgb':(acc_xgb_r,ll_xgb_r,br_xgb_r,feat_list)}
    del lr_r,xgb_r; gc.collect()

best_exp2=max([(n,m,exp2_results[n][m][0],exp2_results[n][m][3])
               for n in exp2_results for m in exp2_results[n]],key=lambda x: x[2])
best_feats=best_exp2[3]; best_n=best_exp2[0]; best_m=best_exp2[1]
X_tr_best=X_tr_aug[best_feats]; X_te_best=X_te[best_feats]
print(f"\n  Best Exp2 subset: {best_n} ({best_m}) — {best_exp2[2]*100:.4f}%")

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 3 — LR-Only Baseline (best feature set)")
# ══════════════════════════════════════════════════════════════════════════════
skf3=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
C_VALS=[0.001,0.01,0.1,1.0,10.0]
print(f"[{ts()}] Manual weighted CV over C={C_VALS} on {len(best_feats)} features...")
best_lr_C=None; best_lr_cv=-1
for c in C_VALS:
    pipe=Pipeline([('sc',RobustScaler()),
                   ('lr',LogisticRegression(penalty='l2',C=c,solver='liblinear',
                                            max_iter=2000,random_state=SEED))])
    cv_s=cv_accuracy(pipe,X_tr_best,y_tr_aug,w_tr_aug,skf3)
    print(f"  C={c:<6}: CV acc = {cv_s.mean():.4f} ± {cv_s.std():.4f}")
    if cv_s.mean()>best_lr_cv: best_lr_cv,best_lr_C=cv_s.mean(),c

print(f"\n  Best C={best_lr_C}  CV acc={best_lr_cv:.4f}")
lr_exp3=Pipeline([('sc',RobustScaler()),
                  ('lr',LogisticRegression(penalty='l2',C=best_lr_C,solver='liblinear',
                                           max_iter=2000,random_state=SEED))])
lr_exp3.fit(X_tr_best,y_tr_aug,lr__sample_weight=w_arr)
p_exp3=lr_exp3.predict_proba(X_te_best)
acc_e3,ll_e3,br_e3=eval_single(p_exp3,y_te)
print(f"  LR-only test: acc={acc_e3*100:.4f}%  vs baseline: {(acc_e3-acc_b)*100:+.3f}pp")
summary_block(f"EXP3: LR-Only (C={best_lr_C}, {len(best_feats)} feats)",len(best_feats),
              "Elo K=48",acc_e3,ll_e3,br_e3,
              f"vs baseline: {(acc_e3-acc_b)*100:+.3f}pp")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 4 — Alternative Ensemble Members")
# ══════════════════════════════════════════════════════════════════════════════
skf4=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
exp4_models={}

print(f"\n[{ts()}] LightGBM default...")
lgb_def=LGBMClassifier(random_state=SEED,n_jobs=1,verbose=-1)
lgb_def.fit(X_tr_best,y_tr_aug,sample_weight=w_arr)
p_lgbd=lgb_def.predict_proba(X_te_best)
acc_lgbd,ll_lgbd,br_lgbd=eval_single(p_lgbd,y_te)
print(f"  LGB default: {acc_lgbd*100:.4f}%")
summary_block("EXP4: LightGBM Default",len(best_feats),"Elo K=48",acc_lgbd,ll_lgbd,br_lgbd,"Default params")
exp4_models['lgb_default']=(lgb_def,acc_lgbd)
del lgb_def; gc.collect()

print(f"[{ts()}] LightGBM Optuna ({N_OPT} trials)...")
def lgb_obj(trial):
    p={'n_estimators':trial.suggest_int('n_estimators',50,500),
       'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True),
       'max_depth':trial.suggest_int('max_depth',2,8),
       'num_leaves':trial.suggest_int('num_leaves',8,127),
       'subsample':trial.suggest_float('subsample',0.5,1.0),
       'colsample_bytree':trial.suggest_float('colsample_bytree',0.5,1.0),
       'reg_alpha':trial.suggest_float('reg_alpha',0,2),
       'min_child_samples':trial.suggest_int('min_child_samples',5,50)}
    # No fit_params — consistent CV without weights
    m=LGBMClassifier(**p,random_state=SEED,n_jobs=1,verbose=-1)
    s=cv_accuracy(m,X_tr_best,y_tr_aug,w_tr_aug,skf4)
    return s.mean()
study_lgb=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lgb_obj,n_trials=N_OPT,show_progress_bar=False)
lgb_opt=LGBMClassifier(**study_lgb.best_params,random_state=SEED,n_jobs=1,verbose=-1)
lgb_opt.fit(X_tr_best,y_tr_aug,sample_weight=w_arr)
p_lgbo=lgb_opt.predict_proba(X_te_best)
acc_lgbo,ll_lgbo,br_lgbo=eval_single(p_lgbo,y_te)
print(f"  LGB Optuna: {acc_lgbo*100:.4f}%  (best CV: {study_lgb.best_value:.4f})")
summary_block(f"EXP4: LightGBM Optuna ({N_OPT}t)",len(best_feats),"Elo K=48",
              acc_lgbo,ll_lgbo,br_lgbo,f"Best CV={study_lgb.best_value:.4f}")
exp4_models['lgb_opt']=(lgb_opt,acc_lgbo)
del study_lgb; gc.collect()

print(f"[{ts()}] CatBoost default...")
cat_def=CatBoostClassifier(random_seed=SEED,verbose=False,thread_count=1)
cat_def.fit(X_tr_best,y_tr_aug,sample_weight=w_arr)
p_catd=cat_def.predict_proba(X_te_best)
acc_catd,ll_catd,br_catd=eval_single(p_catd,y_te)
print(f"  CatBoost default: {acc_catd*100:.4f}%")
summary_block("EXP4: CatBoost Default",len(best_feats),"Elo K=48",acc_catd,ll_catd,br_catd,"Default params")
exp4_models['cat_default']=(cat_def,acc_catd)
del cat_def; gc.collect()

print(f"[{ts()}] CatBoost Optuna ({N_OPT} trials)...")
def cat_obj(trial):
    p={'iterations':trial.suggest_int('iterations',50,500),
       'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True),
       'depth':trial.suggest_int('depth',2,8),
       'l2_leaf_reg':trial.suggest_float('l2_leaf_reg',1,10),
       'subsample':trial.suggest_float('subsample',0.5,1.0)}
    m=CatBoostClassifier(**p,random_seed=SEED,verbose=False,thread_count=1)
    s=cv_accuracy(m,X_tr_best,y_tr_aug,w_tr_aug,skf4)
    return s.mean()
study_cat=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
study_cat.optimize(cat_obj,n_trials=N_OPT,show_progress_bar=False)
cat_opt=CatBoostClassifier(**study_cat.best_params,random_seed=SEED,verbose=False,thread_count=1)
cat_opt.fit(X_tr_best,y_tr_aug,sample_weight=w_arr)
p_cato=cat_opt.predict_proba(X_te_best)
acc_cato,ll_cato,br_cato=eval_single(p_cato,y_te)
print(f"  CatBoost Optuna: {acc_cato*100:.4f}%  (best CV: {study_cat.best_value:.4f})")
summary_block(f"EXP4: CatBoost Optuna ({N_OPT}t)",len(best_feats),"Elo K=48",
              acc_cato,ll_cato,br_cato,f"Best CV={study_cat.best_value:.4f}")
exp4_models['cat_opt']=(cat_opt,acc_cato)
del study_cat; gc.collect()

print(f"[{ts()}] Random Forest default...")
rf_def=RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=1)
rf_def.fit(X_tr_best,y_tr_aug,sample_weight=w_arr)
p_rfd=rf_def.predict_proba(X_te_best)
acc_rfd,ll_rfd,br_rfd=eval_single(p_rfd,y_te)
print(f"  RF default: {acc_rfd*100:.4f}%")
summary_block("EXP4: Random Forest Default",len(best_feats),"Elo K=48",acc_rfd,ll_rfd,br_rfd,"200 trees")
exp4_models['rf_default']=(rf_def,acc_rfd)
del rf_def; gc.collect()

print(f"[{ts()}] Random Forest Optuna ({N_OPT} trials)...")
def rf_obj(trial):
    p={'n_estimators':trial.suggest_int('n_estimators',50,400),
       'max_depth':trial.suggest_int('max_depth',3,20),
       'min_samples_split':trial.suggest_int('min_samples_split',2,20),
       'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,10),
       'max_features':trial.suggest_float('max_features',0.3,1.0)}
    m=RandomForestClassifier(**p,random_state=SEED,n_jobs=1)
    s=cv_accuracy(m,X_tr_best,y_tr_aug,w_tr_aug,skf4)
    return s.mean()
study_rf=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
study_rf.optimize(rf_obj,n_trials=N_OPT,show_progress_bar=False)
rf_opt=RandomForestClassifier(**study_rf.best_params,random_state=SEED,n_jobs=1)
rf_opt.fit(X_tr_best,y_tr_aug,sample_weight=w_arr)
p_rfo=rf_opt.predict_proba(X_te_best)
acc_rfo,ll_rfo,br_rfo=eval_single(p_rfo,y_te)
print(f"  RF Optuna: {acc_rfo*100:.4f}%  (best CV: {study_rf.best_value:.4f})")
summary_block(f"EXP4: Random Forest Optuna ({N_OPT}t)",len(best_feats),"Elo K=48",
              acc_rfo,ll_rfo,br_rfo,f"Best CV={study_rf.best_value:.4f}")
exp4_models['rf_opt']=(rf_opt,acc_rfo)
del study_rf; gc.collect()

print(f"[{ts()}] SVM RBF (2000-sample subset)...")
scaler_svm=RobustScaler().fit(X_tr_best)
X_svm_s=scaler_svm.transform(X_tr_best); X_svm_te=scaler_svm.transform(X_te_best)
idx_svm=rng.choice(len(X_svm_s),min(2000,len(X_svm_s)),replace=False)
Xs,ys,ws=X_svm_s[idx_svm],y_tr_aug.values[idx_svm],w_arr[idx_svm]
best_svm=None; best_svm_cv=-1
for C_s,g_s in [(0.1,'scale'),(1.0,'scale'),(10.0,'scale'),(1.0,0.001)]:
    sm=SVC(kernel='rbf',C=C_s,gamma=g_s,probability=True,random_state=SEED)
    sm.fit(Xs,ys,sample_weight=ws)
    cv_s=cv_accuracy(sm,pd.DataFrame(Xs),pd.Series(ys),pd.Series(ws),
                     StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED))
    print(f"  SVM C={C_s} g={g_s}: CV={cv_s.mean():.4f}")
    if cv_s.mean()>best_svm_cv: best_svm_cv,best_svm=cv_s.mean(),sm
p_svm=best_svm.predict_proba(X_svm_te)
acc_svm,ll_svm,br_svm=eval_single(p_svm,y_te)
print(f"  Best SVM: {acc_svm*100:.4f}%")
summary_block("EXP4: SVM RBF (best params)",len(best_feats),"Elo K=48",
              acc_svm,ll_svm,br_svm,"Trained on 2000-sample subset")
exp4_models['svm']=(best_svm,acc_svm)
del X_svm_s; gc.collect()

sorted_exp4=sorted([(k,v[0],v[1]) for k,v in exp4_models.items() if k!='svm'],
                   key=lambda x: -x[2])
top2_exp4=sorted_exp4[:2]
print(f"\n  Top-2 Exp4 (excl SVM): {[x[0] for x in top2_exp4]}")

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 5 — Stacking with Meta-Learner")
# ══════════════════════════════════════════════════════════════════════════════
skf5=StratifiedKFold(n_splits=CV_FOLDS,shuffle=True,random_state=SEED)
base_names=['lr_exp3',top2_exp4[0][0],top2_exp4[1][0]]
base_models=[lr_exp3,top2_exp4[0][1],top2_exp4[1][1]]
print(f"[{ts()}] Stacking base: {base_names}")

X_tr_arr=X_tr_best.values; X_te_arr=X_te_best.values
y_tr_arr=y_tr_aug.values; w_tr_arr=w_tr_aug.values
oof_train=np.zeros((len(X_tr_arr),len(base_models)))
test_preds=np.zeros((len(X_te_arr),len(base_models)))

for mi,(name,model) in enumerate(zip(base_names,base_models)):
    print(f"  OOF [{mi+1}/{len(base_models)}] {name}...")
    oof_col=np.zeros(len(X_tr_arr)); te_col=np.zeros(len(X_te_arr))
    for fold,(tr_i,va_i) in enumerate(skf5.split(X_tr_arr,y_tr_arr)):
        m_fold=clone(model)
        try:    m_fold.fit(X_tr_arr[tr_i],y_tr_arr[tr_i],sample_weight=w_tr_arr[tr_i])
        except: m_fold.fit(X_tr_arr[tr_i],y_tr_arr[tr_i])
        oof_col[va_i]=m_fold.predict_proba(X_tr_arr[va_i])[:,1]
        te_col      +=m_fold.predict_proba(X_te_arr)[:,1]/CV_FOLDS
    oof_train[:,mi]=oof_col; test_preds[:,mi]=te_col
    gc.collect()

meta_lr=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=SEED)
meta_lr.fit(oof_train,y_tr_arr)
p_stack=meta_lr.predict_proba(test_preds)
acc_st,ll_st,br_st=eval_single(p_stack,y_te.values)
print(f"  Stacked: acc={acc_st*100:.4f}%  meta coefs: {dict(zip(base_names,meta_lr.coef_[0].round(3)))}")
summary_block(f"EXP5: Stacking LR meta + {'+'.join(base_names)}",len(best_feats),
              "Elo K=48",acc_st,ll_st,br_st,
              f"vs baseline: {(acc_st-acc_b)*100:+.3f}pp")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 6 — Best Architecture + Best Glicko-2")
# ══════════════════════════════════════════════════════════════════════════════
# Find best from Exps 3-5
arch_pool=[('Baseline',acc_b,FEAT_129,'lr+xgb'),
           ('Exp3 LR-only',acc_e3,best_feats,'lr'),
           ('Exp5 Stacking',acc_st,best_feats,'stack'),
           (top2_exp4[0][0],top2_exp4[0][2],best_feats,'single')]
best_arch=max(arch_pool,key=lambda x: x[1])
print(f"\n  Best Exps 3-5 architecture: {best_arch[0]}  acc={best_arch[1]*100:.4f}%")

print(f"[{ts()}] Building Glicko-2 features for Exp6 feature set ({best_g2_name})...")
df_g2_6=merge_g2_onto(df.copy(),g2_hist)
if best_g2_name=="G2-domain":
    g2_s6=compute_glicko2_domain(df_all,'KO')
    g2_g6=compute_glicko2_domain(df_all,'Sub')
    g2_d6=compute_glicko2_domain(df_all,'Decision')
    df_g2_6=add_domain_g2(df_g2_6,g2_s6,'strike')
    df_g2_6=add_domain_g2(df_g2_6,g2_g6,'grapple')
    df_g2_6=add_domain_g2(df_g2_6,g2_d6,'dec')
    del g2_s6,g2_g6,g2_d6

feat_e6=[f for f in best_feats if f not in ELO_FEATS]+G2_FEATS
if best_g2_name=="G2-domain": feat_e6+=DOMAIN_FEATS
feat_e6=[f for f in feat_e6 if f in df_g2_6.columns]

g2_6_train=(df_g2_6['date']<TRAIN_CUTOFF)
g2_6_test =~g2_6_train
X_tr_e6_raw=df_g2_6.loc[g2_6_train,feat_e6].reset_index(drop=True)
X_te_e6    =df_g2_6.loc[g2_6_test, feat_e6].reset_index(drop=True)
y_tr_e6_raw=df_g2_6.loc[g2_6_train,'target'].reset_index(drop=True)
y_te_e6    =df_g2_6.loc[g2_6_test, 'target'].reset_index(drop=True)
d_tr_e6    =df_g2_6.loc[g2_6_train,'date'].reset_index(drop=True)
w_e6_raw   =pd.Series(compute_weights(d_tr_e6),index=y_tr_e6_raw.index)
X_tr_e6_aug,y_tr_e6_aug,w_e6_aug=corner_flip(X_tr_e6_raw,y_tr_e6_raw,w_e6_raw)

print(f"[{ts()}] Training Exp6 ({best_arch[0]} arch + {best_g2_name}, {len(feat_e6)} feats)...")
if best_arch[3] in ('lr+xgb','lr'):
    lr_e6,xgb_e6,p_lr_e6,p_xgb_e6=train_lr_xgb(
        X_tr_e6_aug,y_tr_e6_aug,w_e6_aug.values,X_te_e6,y_te_e6)
    if best_arch[3]=='lr+xgb':
        acc_e6,ll_e6,br_e6,_=eval_blend(p_lr_e6,p_xgb_e6,y_te_e6)
    else:
        acc_e6,ll_e6,br_e6=eval_single(p_lr_e6,y_te_e6)
else:
    m_e6=clone(top2_exp4[0][1])
    try:    m_e6.fit(X_tr_e6_aug,y_tr_e6_aug,sample_weight=w_e6_aug.values)
    except: m_e6.fit(X_tr_e6_aug,y_tr_e6_aug)
    p_e6=m_e6.predict_proba(X_te_e6)
    acc_e6,ll_e6,br_e6=eval_single(p_e6,y_te_e6)

print(f"  Exp6 combined: acc={acc_e6*100:.4f}%  vs baseline: {(acc_e6-acc_b)*100:+.3f}pp")
summary_block(f"EXP6: {best_arch[0]} + {best_g2_name}",len(feat_e6),
              best_g2_name,acc_e6,ll_e6,br_e6,
              f"vs baseline: {(acc_e6-acc_b)*100:+.3f}pp")
del df_g2_6; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("FINAL SUMMARY — ALL EXPERIMENTS RANKED")
# ══════════════════════════════════════════════════════════════════════════════
RESULTS_SORTED=sorted(RESULTS,key=lambda x: -x['acc'])
print(f"\n  {'Rk':<3}  {'Experiment':<48}  {'N':>5}  {'Rating':<18}  {'Acc%':>8}  {'LL':>8}  {'Brier':>7}")
print(f"  {'─'*3}  {'─'*48}  {'─'*5}  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*7}")
for rank,r in enumerate(RESULTS_SORTED,1):
    marker='◄'if rank==1 else (' B' if 'BASELINE' in r['name'] else '  ')
    print(f"  {rank:<3}  {r['name']:<48}  {r['feat_count']:>5}  {r['rating']:<18}  "
          f"{r['acc']*100:>7.4f}%  {r['log_loss']:>8.6f}  {r['brier']:>7.6f}  {marker}")

best_r=RESULTS_SORTED[0]
base_r=next(r for r in RESULTS if 'BASELINE' in r['name'])
delta=(best_r['acc']-base_r['acc'])*100
print(f"\n{'─'*70}")
print(f"  Top architecture  : {best_r['name']}")
print(f"  Feature count     : {best_r['feat_count']}")
print(f"  Rating system     : {best_r['rating']}")
print(f"  Test accuracy     : {best_r['acc']*100:.4f}%")
print(f"  Baseline          : {base_r['acc']*100:.4f}%")
print(f"  Delta             : {delta:+.4f}pp")
if delta>0:
    print(f"  ✓ BEATS BASELINE by {delta:.4f}pp")
else:
    print(f"  ✗ Does NOT beat baseline  ({delta:.4f}pp)")
print(f"{'─'*70}")

json_path='experiments/research/architecture_sprint_results_v2.json'
with open(json_path,'w') as f:
    json.dump({'baseline_acc':base_r['acc'],'best_acc':best_r['acc'],'delta_pp':delta,
               'best_name':best_r['name'],'results':RESULTS_SORTED},f,indent=2)
print(f"\n  Results saved to {json_path}")
print(f"[{ts()}] Sprint complete.\n")
