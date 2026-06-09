#!/usr/bin/env python3
"""
architecture_sprint_v3.py — Continuation: Exp 3-6 + final summary.

All Exp 1-2 results hardcoded from v2 run.
Key fix: except (TypeError, ValueError) in cv_accuracy so Pipeline.fit works.
"""
import gc, json, math, os, time, warnings
from collections import defaultdict
from sklearn.base import clone
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
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

SEED=42; TRAIN_START='2015-01-01'; TRAIN_CUTOFF=pd.Timestamp('2024-01-01')
HL_DAYS=730; N_OPT=30; CV_FOLDS=5
WOMENS=["Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight"]

# ─── Known results from v2 run ────────────────────────────────────────────────
RESULTS = [
    dict(name="BASELINE (LR70+XGB30, Elo, 129 feats)",feat_count=129,
         rating="Elo K=48",acc=0.725000,log_loss=0.555988,brier=0.188584,
         note="Reference: 72.50%"),
    dict(name="EXP1A: Glicko-2 Standard (replaces Elo)",feat_count=129,
         rating="Glicko-2 tau=0.5",acc=0.714583,log_loss=0.558450,brier=0.189734,
         note="vs baseline: -1.04pp"),
    dict(name="EXP1B: Glicko-2 + Domain (strike/grapple/dec)",feat_count=132,
         rating="Glicko-2 std+domain",acc=0.721875,log_loss=0.558145,brier=0.189579,
         note="vs baseline: -0.31pp"),
    dict(name="EXP2 LR-Top60",feat_count=60,rating="Elo K=48",
         acc=0.717708,log_loss=0.562580,brier=0.191046,note="LR, 60 feats"),
    dict(name="EXP2 XGB-Top60",feat_count=60,rating="Elo K=48",
         acc=0.698958,log_loss=0.584033,brier=0.199772,note="XGB, 60 feats"),
    dict(name="EXP2 LR-Top40",feat_count=40,rating="Elo K=48",
         acc=0.713542,log_loss=0.566880,brier=0.192875,note="LR, 40 feats"),
    dict(name="EXP2 XGB-Top40",feat_count=40,rating="Elo K=48",
         acc=0.702083,log_loss=0.594175,brier=0.202177,note="XGB, 40 feats"),
    dict(name="EXP2 LR-Top25",feat_count=25,rating="Elo K=48",
         acc=0.715625,log_loss=0.573714,brier=0.195741,note="LR, 25 feats"),
    dict(name="EXP2 XGB-Top25",feat_count=25,rating="Elo K=48",
         acc=0.665625,log_loss=0.596045,brier=0.205296,note="XGB, 25 feats"),
]
BASELINE_ACC = 0.725000
BEST_G2_NAME = "G2-domain"   # best from Exp 1B

def ts(): return time.strftime('%H:%M:%S')
def print_section(t): print(f"\n{'='*70}\n  {t}  [{ts()}]\n{'='*70}")

def summary_block(name,n,rating,acc,ll,br,note=""):
    print(f"\n  ┌{'─'*64}┐")
    print(f"  │  Experiment : {name:<48}│")
    print(f"  │  Features   : {n:<48}│")
    print(f"  │  Rating sys : {rating:<48}│")
    print(f"  │  Test acc   : {acc*100:.4f}%{'':<40}│")
    print(f"  │  Log-loss   : {ll:.6f}{'':<45}│")
    print(f"  │  Brier      : {br:.6f}{'':<45}│")
    print(f"  │  Note       : {str(note):<48}│")
    print(f"  └{'─'*64}┘")
    RESULTS.append(dict(name=name,feat_count=n,rating=rating,acc=acc,
                        log_loss=ll,brier=br,note=str(note)))

def eval_blend(plr,px,yt,w1=0.70,w2=0.30):
    p=w1*plr+w2*px
    return accuracy_score(yt,(p[:,1]>0.5).astype(int)),log_loss(yt,p),brier_score_loss(yt,p[:,1]),p

def eval_single(p,yt):
    return accuracy_score(yt,(p[:,1]>0.5).astype(int)),log_loss(yt,p),brier_score_loss(yt,p[:,1])

# ── Fixed cv_accuracy: catches both TypeError AND ValueError ──────────────────
def cv_accuracy(model, X, y, w, skf):
    X_a=X.values if hasattr(X,'values') else X
    y_a=y.values if hasattr(y,'values') else y
    w_a=w.values if hasattr(w,'values') else w
    scores=[]
    for tr_i,va_i in skf.split(X_a,y_a):
        m=clone(model)
        try:
            m.fit(X_a[tr_i],y_a[tr_i],sample_weight=w_a[tr_i])
        except (TypeError,ValueError):
            m.fit(X_a[tr_i],y_a[tr_i])          # Pipeline or unsupported → no weights
        scores.append(accuracy_score(y_a[va_i],m.predict(X_a[va_i])))
    return np.array(scores)

def corner_flip(X,y,w=None):
    Xf=X.copy()
    for c in list(X.columns):
        if c.startswith('R_'):
            bc='B_'+c[2:]
            if bc in X.columns: Xf[c],Xf[bc]=X[bc].values,X[c].values
    for c in Xf.columns:
        if c.endswith('_dif'): Xf[c]=-Xf[c]
    Xa=pd.concat([X,Xf],ignore_index=True); ya=pd.concat([y,1-y],ignore_index=True)
    if w is not None:
        wa=pd.concat([w,w],ignore_index=True); return Xa,ya,wa
    return Xa,ya

def compute_weights(dates):
    return np.exp(-np.log(2)*(TRAIN_CUTOFF-dates).dt.days.clip(lower=0)/HL_DAYS)

XGB_P={'n_estimators':200,'learning_rate':0.1,'max_depth':6,'min_child_weight':1,
       'subsample':0.8,'colsample_bytree':0.7,'gamma':0.3,'reg_alpha':0,'reg_lambda':2.0,
       'random_state':SEED,'eval_metric':'logloss','verbosity':0,'n_jobs':1}
LR_C=0.00711

def train_lr_xgb(Xtr,ytr,wa,Xte,yte):
    w=wa.values if hasattr(wa,'values') else wa
    lr=Pipeline([('sc',RobustScaler()),
                 ('lr',LogisticRegression(penalty='l2',C=LR_C,solver='liblinear',
                                          max_iter=2000,random_state=SEED))])
    lr.fit(Xtr,ytr,lr__sample_weight=w)
    xgb=XGBClassifier(**XGB_P); xgb.fit(Xtr,ytr,sample_weight=w)
    return lr,xgb,lr.predict_proba(Xte),xgb.predict_proba(Xte)

# ─────────────────────────────────────────────────────────────────────────────
print_section("SHARED SETUP — Data Loading + Feature Engineering")
# ─────────────────────────────────────────────────────────────────────────────
print(f"[{ts()}] Loading data...")
df_all=pd.read_csv('data/ufc-master.csv',low_memory=False)
df_all['date']=pd.to_datetime(df_all['date'])
career_raw=pd.read_csv('data/career_fights_updated.csv')
career_raw['date']=pd.to_datetime(career_raw['date'])
career_raw=career_raw.sort_values(['fighter','date']).reset_index(drop=True)
style_df=pd.read_csv('data/ufc_fighters_final_updated.csv')
for c in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    style_df[c]=pd.to_numeric(style_df[c].astype(str).str.replace('%','',regex=False),
                               errors='coerce').fillna(0.0)/100.0

print(f"[{ts()}] Computing Elo K=48...")
def elo_hist(df_src,K=48,base=1500.0):
    ds=df_src.sort_values('date').reset_index(drop=True); elo={}; rows=[]
    for _,row in ds.iterrows():
        r,b=row['R_fighter'],row['B_fighter']
        rb,bb=elo.get(r,base),elo.get(b,base)
        re=1.0/(1.0+10.0**((bb-rb)/400.0))
        ra=1.0 if row['Winner']=='Red' else(0.0 if row['Winner']=='Blue' else 0.5)
        rows+=[{'fighter':r,'date':row['date'],'elo_before':rb,'elo_after':rb+K*(ra-re)},
               {'fighter':b,'date':row['date'],'elo_before':bb,'elo_after':bb+K*((1-ra)-(1-re))}]
        elo[r]=rb+K*(ra-re); elo[b]=bb+K*((1-ra)-(1-re))
    h=pd.DataFrame(rows).sort_values(['fighter','date']).reset_index(drop=True)
    h['elo_trend']=h.groupby('fighter')['elo_before'].transform(lambda x: x-x.shift(3))
    return h
eh=elo_hist(df_all)

print(f"[{ts()}] Career stats...")
def career_stats(cdf):
    df=cdf.sort_values(['fighter','date']).copy().reset_index(drop=True)
    df['_ko']=((df['won']==1)&df['method'].str.contains('KO|TKO',case=False,na=False)).astype(float)
    df['_sub']=((df['won']==1)&df['method'].str.contains('Sub|Submission',case=False,na=False)).astype(float)
    df['_fin']=((df['won']==1)&df['method'].str.contains('KO|TKO|Sub',case=False,na=False)).astype(float)
    g=df.groupby('fighter',sort=False); df['cum_fights']=g.cumcount()
    for s,d in [('won','_cs_won'),('_ko','_cs_ko'),('_sub','_cs_sub'),('_fin','_cs_fin')]:
        df[d]=g[s].cumsum()-df[s]
    sn=df['cum_fights'].clip(lower=1)
    df['career_win_rate']=np.where(df['cum_fights']>0,df['_cs_won']/sn,0.5)
    df['ko_finish_rate']=np.where(df['cum_fights']>0,df['_cs_ko']/sn,0.0)
    df['sub_finish_rate']=np.where(df['cum_fights']>0,df['_cs_sub']/sn,0.0)
    df['career_finish_rate']=np.where(df['cum_fights']>0,df['_cs_fin']/sn,0.0)
    def _r(s,w,d): return s.shift(1).rolling(w,min_periods=1).mean().fillna(d)
    df['last3_win_rate']=g['won'].transform(lambda x:_r(x,3,0.5))
    df['last10_win_rate']=g['won'].transform(lambda x:_r(x,10,0.5))
    df['last5_won']=g['won'].transform(lambda x:_r(x,5,0.5))
    df['last5_finish_rate']=g['_fin'].transform(lambda x:_r(x,5,0.0))
    df['trend_score']=df['last3_win_rate']-df['last10_win_rate']
    df['_prev_date']=g['date'].transform(lambda x:x.shift(1))
    df['layoff_days']=(df['date']-df['_prev_date']).dt.days.fillna(180.0).clip(lower=0)
    all_wr={f:grp['won'].sum()/max(1,len(grp)) for f,grp in df.groupby('fighter')}
    opp_col=df['opponent'].tolist(); ftr_col=df['fighter'].tolist()
    ftr_pos=defaultdict(list)
    for pos,idx in enumerate(df.index.tolist()): ftr_pos[ftr_col[pos]].append(pos)
    oq=np.full(len(df),0.5)
    for ftr,positions in ftr_pos.items():
        for rank,pos in enumerate(positions):
            past=[opp_col[p] for p in positions[max(0,rank-5):rank]]
            rates=[all_wr.get(o,0.5) for o in past]
            oq[pos]=float(np.mean(rates)) if rates else 0.5
    df['opp_quality']=oq
    return df[['fighter','date','cum_fights','career_win_rate','ko_finish_rate',
               'sub_finish_rate','career_finish_rate','last3_win_rate','last10_win_rate',
               'last5_won','last5_finish_rate','trend_score','layoff_days','opp_quality']]
cs=career_stats(career_raw)

print(f"[{ts()}] QA stats...")
def qa_stats(cdf,eh):
    od=cdf[['fighter','opponent','date','won','got_finish']].copy().rename(columns={'opponent':'opp_name'})
    oref=eh[['fighter','date','elo_before']].rename(columns={'fighter':'opp_name','elo_before':'opp_elo'})
    od=pd.merge_asof(od.sort_values('date'),oref.sort_values('date'),on='date',by='opp_name',direction='backward')
    od['opp_elo']=od['opp_elo'].fillna(1500.0); od['ew']=od['opp_elo']/1500.0
    rows=[]
    for fighter,grp in od.groupby('fighter',sort=False):
        grp=grp.sort_values('date'); n=len(grp)
        qw=np.full(n,0.5); qf=np.full(n,0.0); ql=np.full(n,0.0); qa=np.full(n,0.0)
        cew=ceww=cewf=cn=co=cd=0.0
        for i,(_,r) in enumerate(grp.iterrows()):
            if cew>0: qw[i]=ceww/cew; qf[i]=cewf/cew
            if cn>0: ql[i]=co/cn; qa[i]=cd/cn
            ew=r['ew']; w=r['won']; f=r['got_finish'] if pd.notna(r.get('got_finish')) else 0.0
            cew+=ew; ceww+=ew*w; cewf+=ew*f; cn+=ew; co+=ew*w; cd+=ew*(1.0-w)
        rows.append(pd.DataFrame({'fighter':fighter,'date':grp['date'].values,
                                   'qa_win_rate':qw,'qa_finish_rate':qf,'qa_SLpM':ql,'qa_SApM':qa}))
    return pd.concat(rows,ignore_index=True).sort_values(['fighter','date'])
qs=qa_stats(career_raw,eh)

WC_ORDER={"Women's Strawweight":0,"Women's Flyweight":1,"Women's Bantamweight":2,
          "Women's Featherweight":3,"Flyweight":4,"Bantamweight":5,"Featherweight":6,
          "Lightweight":7,"Welterweight":8,"Middleweight":9,"Light Heavyweight":10,
          "Heavyweight":11,"Catch Weight":6}

print(f"[{ts()}] Building fight DataFrame + merging all stats...")
df=df_all[(df_all['date']>=TRAIN_START)&df_all['Winner'].isin(['Red','Blue'])&
          ~df_all['weight_class'].isin(WOMENS)].copy().sort_values('date').reset_index(drop=True)

cc=[c for c in cs.columns if c not in ('fighter','date')]
for pf,pfx in [('R_fighter',{c:f'R_{c}' for c in cc}),('B_fighter',{c:f'B_{c}' for c in cc})]:
    renamed=cs.rename(columns={'fighter':pf,**pfx})
    df=pd.merge_asof(df.sort_values('date'),renamed.sort_values('date'),on='date',by=pf,direction='backward')
for stat,dv in {'cum_fights':0,'career_win_rate':0.5,'ko_finish_rate':0.0,'sub_finish_rate':0.0,
               'career_finish_rate':0.0,'last3_win_rate':0.5,'last10_win_rate':0.5,'last5_won':0.5,
               'last5_finish_rate':0.0,'trend_score':0.0,'layoff_days':180.0,'opp_quality':0.5}.items():
    df[f'R_{stat}']=df[f'R_{stat}'].fillna(dv); df[f'B_{stat}']=df[f'B_{stat}'].fillna(dv)
for pf,pfx in [('R_fighter',{'qa_win_rate':'R_qa_win_rate','qa_finish_rate':'R_qa_finish_rate',
                              'qa_SLpM':'R_qa_SLpM','qa_SApM':'R_qa_SApM'}),
               ('B_fighter',{'qa_win_rate':'B_qa_win_rate','qa_finish_rate':'B_qa_finish_rate',
                              'qa_SLpM':'B_qa_SLpM','qa_SApM':'B_qa_SApM'})]:
    df=pd.merge_asof(df.sort_values('date'),qs.rename(columns={'fighter':pf,**pfx}).sort_values('date'),
                     on='date',by=pf,direction='backward')
for c in ['R_qa_win_rate','R_qa_finish_rate','R_qa_SLpM','R_qa_SApM',
          'B_qa_win_rate','B_qa_finish_rate','B_qa_SLpM','B_qa_SApM']:
    df[c]=df[c].fillna(0.5 if 'win_rate' in c else 0.0)
df['qa_win_rate_dif']=df['R_qa_win_rate']-df['B_qa_win_rate']
df['qa_finish_rate_dif']=df['R_qa_finish_rate']-df['B_qa_finish_rate']
df['qa_SLpM_dif']=df['R_qa_SLpM']-df['B_qa_SLpM']
df['qa_SApM_dif']=df['R_qa_SApM']-df['B_qa_SApM']
for pf,sfx in [('R_fighter','R_'),('B_fighter','B_')]:
    df=pd.merge_asof(df.sort_values('date'),
                     eh[['fighter','date','elo_before','elo_trend']].rename(
                         columns={'fighter':pf,'elo_before':f'{sfx}elo','elo_trend':f'{sfx}elo_trend'}
                     ).sort_values('date'),on='date',by=pf,direction='backward')
for c in ['R_elo','B_elo']: df[c]=df[c].fillna(1500.0)
for c in ['R_elo_trend','B_elo_trend']: df[c]=df[c].fillna(0.0)
df['elo_dif']=df['R_elo']-df['B_elo']; df['elo_trend_dif']=df['R_elo_trend']-df['B_elo_trend']
style_src=['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']
sdf=style_df.drop_duplicates(subset=['Fighter_Name'],keep='last')
for pf,pfx in [('R_fighter',{s:f'R_{s}' for s in style_src}),('B_fighter',{s:f'B_{s}' for s in style_src})]:
    df=df.merge(sdf[['Fighter_Name']+style_src].rename(columns={'Fighter_Name':pf,**pfx}),on=pf,how='left')
for c in [f'{p}{s}' for p in('R_','B_') for s in style_src]:
    df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
df['weight_class_ord']=df['weight_class'].map(WC_ORDER).fillna(6).astype(int)
df['R_southpaw']=(df['R_Stance'].str.lower()=='southpaw').astype(int)
df['B_southpaw']=(df['B_Stance'].str.lower()=='southpaw').astype(int)
df['orth_clash']=((df['R_southpaw']==0)&(df['B_southpaw']==0)).astype(int)
df['south_clash']=((df['R_southpaw']==1)&(df['B_southpaw']==1)).astype(int)
df['R_age']=pd.to_numeric(df['R_age'],errors='coerce').fillna(28.0)
df['B_age']=pd.to_numeric(df['B_age'],errors='coerce').fillna(28.0)
df['R_age_x_exp']=df['R_age']*df['R_cum_fights']; df['B_age_x_exp']=df['B_age']*df['B_cum_fights']
df['age_x_exp_dif']=df['R_age_x_exp']-df['B_age_x_exp']
for p,s in [('R_',df['R_layoff_days']),('B_',df['B_layoff_days'])]:
    d=s.fillna(180.0)
    for sfx,cond in [('lt90',d<90),('90_180',(d>=90)&(d<180)),('180_365',(d>=180)&(d<365)),('gt365',d>=365)]:
        df[f'{p}layoff_{sfx}']=cond.astype(int)
for stat in ['career_win_rate','last5_won','last5_finish_rate','opp_quality','trend_score',
             'ko_finish_rate','sub_finish_rate','last3_win_rate','last10_win_rate']:
    df[f'{stat}_dif']=df[f'R_{stat}']-df[f'B_{stat}']
for ra,rb,rd in [('SLpM','SLpM','SLpM_dif'),('SApM','SApM','SApM_dif'),('Str_Def','Str_Def','Str_Def_dif'),
                 ('TD_Def','TD_Def','TD_Def_dif'),('Sub_Avg','Sub_Avg','Sub_Avg_dif'),('TD_Avg','TD_Avg','TD_Avg_dif')]:
    df[rd]=df[f'R_{ra}']-df[f'B_{rb}']
for c in ['R_wins','R_losses','B_wins','B_losses','R_Height_cms','R_Reach_cms','B_Height_cms','B_Reach_cms',
          'R_avg_SIG_STR_landed','R_avg_TD_landed','R_avg_SIG_STR_pct','R_avg_SUB_ATT','R_avg_TD_pct',
          'B_avg_SIG_STR_landed','B_avg_TD_landed','B_avg_SIG_STR_pct','B_avg_SUB_ATT','B_avg_TD_pct',
          'R_current_win_streak','R_current_lose_streak','R_longest_win_streak',
          'B_current_win_streak','B_current_lose_streak','B_longest_win_streak','B_total_title_bouts']:
    df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
df['win_dif']=df['R_wins']-df['B_wins']; df['loss_dif']=df['R_losses']-df['B_losses']
df['win_streak_dif']=df['R_current_win_streak']-df['B_current_win_streak']
df['lose_streak_dif']=df['R_current_lose_streak']-df['B_current_lose_streak']
df['height_dif']=df['R_Height_cms']-df['B_Height_cms']
df['reach_dif']=df['R_Reach_cms']-df['B_Reach_cms']
df['age_dif']=df['R_age']-df['B_age']
df['sig_str_dif']=df['R_avg_SIG_STR_landed']-df['B_avg_SIG_STR_landed']
df['avg_td_dif']=df['R_avg_TD_landed']-df['B_avg_TD_landed']
df['ko_dif']=df['R_ko_finish_rate']-df['B_ko_finish_rate']
df['sub_dif']=df['R_sub_finish_rate']-df['B_sub_finish_rate']
df['total_title_bout_dif']=pd.to_numeric(df.get('R_total_title_bouts',0),errors='coerce').fillna(0)-\
                           pd.to_numeric(df.get('B_total_title_bouts',0),errors='coerce').fillna(0)
cdf2=career_raw.sort_values(['fighter','date']).copy()
cdf2['is_loss']=(cdf2['won']==0).astype(float)
cdf2['is_fin_loss']=((cdf2['won']==0)&(cdf2['got_finish'].fillna(0)==1)).astype(float)
g2r=cdf2.groupby('fighter',sort=False)
cdf2['_cs_l']=g2r['is_loss'].cumsum()-cdf2['is_loss']
cdf2['_cs_fl']=g2r['is_fin_loss'].cumsum()-cdf2['is_fin_loss']
cdf2['got_finished_rate']=np.where(cdf2['_cs_l']>0,cdf2['_cs_fl']/cdf2['_cs_l'],0.5)
for pf,col in [('R_fighter','R_got_finished_rate'),('B_fighter','B_got_finished_rate')]:
    chin=cdf2[['fighter','date','got_finished_rate']].rename(columns={'fighter':pf,'got_finished_rate':col})
    df=pd.merge_asof(df.sort_values('date'),chin.sort_values('date'),on='date',by=pf,direction='backward')
    df[col]=df[col].fillna(0.5)
df['R_age_x_layoff']=df['R_age']*df['R_layoff_days'].clip(upper=730)
df['B_age_x_layoff']=df['B_age']*df['B_layoff_days'].clip(upper=730)
df['age_x_layoff_dif']=df['R_age_x_layoff']-df['B_age_x_layoff']
df['R_finish_danger']=df['R_ko_finish_rate']+df['R_sub_finish_rate']
df['B_finish_danger']=df['B_ko_finish_rate']+df['B_sub_finish_rate']
df['finish_danger_mismatch']=(df['R_finish_danger']*(1-df['B_got_finished_rate'])-
                              df['B_finish_danger']*(1-df['R_got_finished_rate']))
del cdf2,g2r; gc.collect()

df=df[(df['R_cum_fights']>=1)&(df['B_cum_fights']>=1)].copy().sort_values('date').reset_index(drop=True)
df['target']=(df['Winner']=='Red').astype(int)
print(f"  Fights: {len(df):,}  train={(df['date']<TRAIN_CUTOFF).sum()}  test={(df['date']>=TRAIN_CUTOFF).sum()}")

FEAT_BASE=["R_wins","R_losses","R_Height_cms","R_age","R_avg_SIG_STR_landed","R_avg_TD_landed",
           "R_current_win_streak","R_current_lose_streak","R_longest_win_streak",
           "R_avg_SIG_STR_pct","R_avg_SUB_ATT","R_avg_TD_pct","R_Reach_cms",
           "B_wins","B_losses","B_Height_cms","B_age","B_avg_SIG_STR_landed","B_avg_TD_landed",
           "B_current_win_streak","B_current_lose_streak","B_longest_win_streak",
           "B_avg_SIG_STR_pct","B_avg_SUB_ATT","B_avg_TD_pct","B_Reach_cms","B_total_title_bouts",
           "win_dif","loss_dif","win_streak_dif","lose_streak_dif","height_dif","reach_dif",
           "age_dif","sig_str_dif","avg_td_dif","ko_dif","sub_dif","total_title_bout_dif",
           "weight_class_ord","orth_clash","south_clash","R_southpaw","R_cum_fights","B_cum_fights",
           "R_career_win_rate","B_career_win_rate","career_win_rate_dif","R_last5_won","B_last5_won",
           "last5_won_dif","R_last5_finish_rate","B_last5_finish_rate","last5_finish_rate_dif",
           "R_opp_quality","B_opp_quality","opp_quality_dif","R_trend_score","B_trend_score",
           "trend_score_dif","R_ko_finish_rate","B_ko_finish_rate","ko_finish_rate_dif",
           "R_sub_finish_rate","B_sub_finish_rate","sub_finish_rate_dif","R_last3_win_rate",
           "B_last3_win_rate","last3_win_rate_dif","R_last10_win_rate","B_last10_win_rate",
           "R_age_x_exp","B_age_x_exp","age_x_exp_dif","R_layoff_lt90","R_layoff_90_180",
           "R_layoff_180_365","R_layoff_gt365","B_layoff_lt90","B_layoff_90_180","B_layoff_180_365",
           "R_SLpM","R_SApM","R_Str_Acc","R_Str_Def","R_TD_Avg","R_TD_Acc","R_TD_Def","R_Sub_Avg",
           "B_SLpM","B_SApM","B_Str_Acc","B_Str_Def","B_TD_Avg","B_TD_Acc","B_TD_Def","B_Sub_Avg",
           "SLpM_dif","SApM_dif","Str_Def_dif","TD_Def_dif","Sub_Avg_dif","TD_Avg_dif",
           "R_elo","B_elo","elo_dif","R_elo_trend","B_elo_trend","elo_trend_dif"]
FEAT_QA=["R_qa_win_rate","R_qa_finish_rate","R_qa_SLpM","R_qa_SApM","B_qa_win_rate",
         "B_qa_finish_rate","B_qa_SLpM","B_qa_SApM","qa_win_rate_dif","qa_finish_rate_dif",
         "qa_SLpM_dif","qa_SApM_dif"]
FEAT_INT=["R_age_x_layoff","B_age_x_layoff","age_x_layoff_dif","R_finish_danger","B_finish_danger",
          "finish_danger_mismatch","R_got_finished_rate","B_got_finished_rate"]
FEAT_129=FEAT_BASE+FEAT_QA+FEAT_INT
for c in FEAT_129: df[c]=pd.to_numeric(df.get(c,0),errors='coerce').fillna(0.0)

train_mask=df['date']<TRAIN_CUTOFF; test_mask=~train_mask
Xtr_r=df.loc[train_mask,FEAT_129].reset_index(drop=True)
ytr_r=df.loc[train_mask,'target'].reset_index(drop=True)
dtr_r=df.loc[train_mask,'date'].reset_index(drop=True)
Xte=df.loc[test_mask,FEAT_129].reset_index(drop=True)
yte=df.loc[test_mask,'target'].reset_index(drop=True)
wr=pd.Series(compute_weights(dtr_r),index=ytr_r.index)
Xtr,ytr,wtr=corner_flip(Xtr_r,ytr_r,wr)
warr=wtr.values
print(f"  Train aug: {len(Xtr):,}  Test: {len(Xte):,}  Feats: {len(FEAT_129)}")
gc.collect()

# Recompute baseline for XGB (needed by SHAP)
print(f"[{ts()}] Recomputing baseline LR+XGB (for SHAP)...")
lr_base,xgb_base,plr_b,pxgb_b=train_lr_xgb(Xtr,ytr,warr,Xte,yte)
acc_b,ll_b,br_b,_=eval_blend(plr_b,pxgb_b,yte)
print(f"  Baseline: {acc_b*100:.4f}%  (stored: {BASELINE_ACC*100:.4f}%)")

# SHAP top-60 features
print(f"[{ts()}] SHAP top-60 features...")
rng=np.random.RandomState(SEED)
shap_idx=rng.choice(len(Xtr),min(400,len(Xtr)),replace=False)
exp=shap.TreeExplainer(xgb_base)
sv=exp.shap_values(Xtr.iloc[shap_idx])
mas=np.abs(sv).mean(axis=0)
ranking=sorted(zip(FEAT_129,mas),key=lambda x:-x[1])
print("  Top 15 features:")
for i,(f,v) in enumerate(ranking[:15],1): print(f"    {i:2d}. {f:<35}: {v:.5f}")
top60=[f for f,_ in ranking[:60]]
top40=[f for f,_ in ranking[:40]]
top25=[f for f,_ in ranking[:25]]
raw_flags=[f for f,_ in ranking[:25] if f in ['R_wins','R_losses','B_wins','B_losses',
           'R_Height_cms','R_Reach_cms','B_Height_cms','B_Reach_cms']]
if raw_flags: print(f"  Raw-stat candidates for ratio replacement: {raw_flags}")
del exp,sv; gc.collect()

# Best feature set = Top60 (matches v2 result)
best_feats=top60
Xtr_best=Xtr[best_feats]; Xte_best=Xte[best_feats]
print(f"  Using Top60 features ({len(best_feats)} feats) for Exp 3-5")

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 3 — LR-Only Baseline (Top60 features)")
# ══════════════════════════════════════════════════════════════════════════════
skf3=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
C_VALS=[0.001,0.01,0.1,1.0,10.0]
print(f"[{ts()}] Weighted CV over C={C_VALS} on 60 features...")
best_C=None; best_cv=-1
for c in C_VALS:
    pipe=Pipeline([('sc',RobustScaler()),
                   ('lr',LogisticRegression(penalty='l2',C=c,solver='liblinear',
                                            max_iter=2000,random_state=SEED))])
    cv_s=cv_accuracy(pipe,Xtr_best,ytr,wtr,skf3)
    print(f"  C={c:<6}: CV={cv_s.mean():.4f} ± {cv_s.std():.4f}")
    if cv_s.mean()>best_cv: best_cv,best_C=cv_s.mean(),c
print(f"\n  Best C={best_C}  CV={best_cv:.4f}")
lr3=Pipeline([('sc',RobustScaler()),
              ('lr',LogisticRegression(penalty='l2',C=best_C,solver='liblinear',
                                       max_iter=2000,random_state=SEED))])
lr3.fit(Xtr_best,ytr,lr__sample_weight=warr)
p3=lr3.predict_proba(Xte_best)
acc3,ll3,br3=eval_single(p3,yte)
print(f"  LR-only: {acc3*100:.4f}%  vs baseline: {(acc3-BASELINE_ACC)*100:+.3f}pp")
summary_block(f"EXP3: LR-Only C={best_C}, Top60",60,"Elo K=48",acc3,ll3,br3,
              f"vs baseline: {(acc3-BASELINE_ACC)*100:+.3f}pp")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 4 — Alternative Ensemble Members")
# ══════════════════════════════════════════════════════════════════════════════
skf4=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
exp4={}

def run_model(name, model, label="Default"):
    try: model.fit(Xtr_best,ytr,sample_weight=warr)
    except TypeError: model.fit(Xtr_best,ytr)
    p=model.predict_proba(Xte_best)
    a,ll,br=eval_single(p,yte)
    print(f"  {name}: {a*100:.4f}%  vs baseline: {(a-BASELINE_ACC)*100:+.3f}pp")
    summary_block(f"EXP4: {name}",60,"Elo K=48",a,ll,br,label)
    return model,a,p

print(f"\n[{ts()}] LightGBM default...")
lgb_def,acc_lgbd,_=run_model("LightGBM Default",LGBMClassifier(random_state=SEED,n_jobs=1,verbose=-1))
exp4['lgb_default']=(lgb_def,acc_lgbd); del lgb_def; gc.collect()

print(f"[{ts()}] LightGBM Optuna ({N_OPT} trials)...")
def lgb_obj(trial):
    m=LGBMClassifier(n_estimators=trial.suggest_int('n',50,500),
                     learning_rate=trial.suggest_float('lr',0.01,0.3,log=True),
                     max_depth=trial.suggest_int('d',2,8),
                     num_leaves=trial.suggest_int('nl',8,127),
                     subsample=trial.suggest_float('ss',0.5,1.0),
                     colsample_bytree=trial.suggest_float('cs',0.5,1.0),
                     reg_alpha=trial.suggest_float('ra',0,2),
                     min_child_samples=trial.suggest_int('mcs',5,50),
                     random_state=SEED,n_jobs=1,verbose=-1)
    return cv_accuracy(m,Xtr_best,ytr,wtr,skf4).mean()
slgb=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
slgb.optimize(lgb_obj,n_trials=N_OPT,show_progress_bar=False)
lgbo=LGBMClassifier(**{k.replace('n','n_estimators').replace('lr','learning_rate')
                        .replace('d','max_depth') if k in ('n','lr','d') else
                        k.replace('nl','num_leaves').replace('ss','subsample')
                        .replace('cs','colsample_bytree').replace('ra','reg_alpha')
                        .replace('mcs','min_child_samples'): v
                        for k,v in slgb.best_params.items()},
                    random_state=SEED,n_jobs=1,verbose=-1)
lgbo,acc_lgbo,_=run_model("LightGBM Optuna",lgbo,f"Best CV={slgb.best_value:.4f}")
exp4['lgb_opt']=(lgbo,acc_lgbo); del slgb; gc.collect()

print(f"[{ts()}] CatBoost default...")
cat_def,acc_catd,_=run_model("CatBoost Default",
                              CatBoostClassifier(random_seed=SEED,verbose=False,thread_count=1))
exp4['cat_default']=(cat_def,acc_catd); del cat_def; gc.collect()

print(f"[{ts()}] CatBoost Optuna ({N_OPT} trials)...")
def cat_obj(trial):
    m=CatBoostClassifier(iterations=trial.suggest_int('it',50,500),
                         learning_rate=trial.suggest_float('lr',0.01,0.3,log=True),
                         depth=trial.suggest_int('d',2,8),
                         l2_leaf_reg=trial.suggest_float('l2',1,10),
                         subsample=trial.suggest_float('ss',0.5,1.0),
                         random_seed=SEED,verbose=False,thread_count=1)
    return cv_accuracy(m,Xtr_best,ytr,wtr,skf4).mean()
scat=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
scat.optimize(cat_obj,n_trials=N_OPT,show_progress_bar=False)
catp=scat.best_params
cato=CatBoostClassifier(iterations=catp['it'],learning_rate=catp['lr'],depth=catp['d'],
                         l2_leaf_reg=catp['l2'],subsample=catp['ss'],
                         random_seed=SEED,verbose=False,thread_count=1)
cato,acc_cato,_=run_model("CatBoost Optuna",cato,f"Best CV={scat.best_value:.4f}")
exp4['cat_opt']=(cato,acc_cato); del scat; gc.collect()

print(f"[{ts()}] Random Forest default...")
rf_def,acc_rfd,_=run_model("Random Forest Default",
                             RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=1))
exp4['rf_default']=(rf_def,acc_rfd); del rf_def; gc.collect()

print(f"[{ts()}] Random Forest Optuna ({N_OPT} trials)...")
def rf_obj(trial):
    m=RandomForestClassifier(n_estimators=trial.suggest_int('n',50,400),
                              max_depth=trial.suggest_int('d',3,20),
                              min_samples_split=trial.suggest_int('mss',2,20),
                              min_samples_leaf=trial.suggest_int('msl',1,10),
                              max_features=trial.suggest_float('mf',0.3,1.0),
                              random_state=SEED,n_jobs=1)
    return cv_accuracy(m,Xtr_best,ytr,wtr,skf4).mean()
srf=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
srf.optimize(rf_obj,n_trials=N_OPT,show_progress_bar=False)
rfp=srf.best_params
rfo=RandomForestClassifier(n_estimators=rfp['n'],max_depth=rfp['d'],
                            min_samples_split=rfp['mss'],min_samples_leaf=rfp['msl'],
                            max_features=rfp['mf'],random_state=SEED,n_jobs=1)
rfo,acc_rfo,_=run_model("Random Forest Optuna",rfo,f"Best CV={srf.best_value:.4f}")
exp4['rf_opt']=(rfo,acc_rfo); del srf; gc.collect()

print(f"[{ts()}] SVM RBF (2000-sample subset)...")
sc_svm=RobustScaler().fit(Xtr_best)
Xs_all=sc_svm.transform(Xtr_best); Xte_svm=sc_svm.transform(Xte_best)
idx_svm=rng.choice(len(Xs_all),min(2000,len(Xs_all)),replace=False)
Xs,ys,ws_svm=Xs_all[idx_svm],ytr.values[idx_svm],warr[idx_svm]
best_svm=None; best_svm_cv=-1
for Cs,gs in [(0.1,'scale'),(1.0,'scale'),(10.0,'scale'),(1.0,0.001)]:
    sm=SVC(kernel='rbf',C=Cs,gamma=gs,probability=True,random_state=SEED)
    sm.fit(Xs,ys,sample_weight=ws_svm)
    cv_s=cv_accuracy(sm,pd.DataFrame(Xs),pd.Series(ys),pd.Series(ws_svm),
                     StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED))
    print(f"  SVM C={Cs} g={gs}: CV={cv_s.mean():.4f}")
    if cv_s.mean()>best_svm_cv: best_svm_cv,best_svm=cv_s.mean(),sm
p_svm=best_svm.predict_proba(Xte_svm)
acc_svm,ll_svm,br_svm=eval_single(p_svm,yte)
print(f"  SVM: {acc_svm*100:.4f}%")
summary_block("EXP4: SVM RBF (best params)",60,"Elo K=48",acc_svm,ll_svm,br_svm,
              "2000-sample subset train")
exp4['svm']=(best_svm,acc_svm); del Xs_all; gc.collect()

sorted_exp4=sorted([(k,v[0],v[1]) for k,v in exp4.items() if k!='svm'],key=lambda x:-x[2])
top2=sorted_exp4[:2]
print(f"\n  Top-2 for stacking: {[x[0] for x in top2]}")

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 5 — Stacking with Meta-Learner")
# ══════════════════════════════════════════════════════════════════════════════
skf5=StratifiedKFold(n_splits=CV_FOLDS,shuffle=True,random_state=SEED)
base_names=['lr_exp3',top2[0][0],top2[1][0]]
base_models=[lr3,top2[0][1],top2[1][1]]
print(f"[{ts()}] OOF for stacking base models: {base_names}")

Xtr_a=Xtr_best.values; Xte_a=Xte_best.values
ytr_a=ytr.values; warr_a=wtr.values
oof=np.zeros((len(Xtr_a),len(base_models)))
tp=np.zeros((len(Xte_a),len(base_models)))

for mi,(nm,m) in enumerate(zip(base_names,base_models)):
    print(f"  OOF [{mi+1}/{len(base_models)}] {nm}...")
    oc=np.zeros(len(Xtr_a)); tc=np.zeros(len(Xte_a))
    for fold,(tri,vai) in enumerate(skf5.split(Xtr_a,ytr_a)):
        mf=clone(m)
        try:    mf.fit(Xtr_a[tri],ytr_a[tri],sample_weight=warr_a[tri])
        except (TypeError,ValueError): mf.fit(Xtr_a[tri],ytr_a[tri])
        oc[vai]=mf.predict_proba(Xtr_a[vai])[:,1]
        tc+=mf.predict_proba(Xte_a)[:,1]/CV_FOLDS
    oof[:,mi]=oc; tp[:,mi]=tc; gc.collect()

meta=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=SEED)
meta.fit(oof,ytr_a)
p_st=meta.predict_proba(tp)
acc_st,ll_st,br_st=eval_single(p_st,yte.values)
print(f"  Stack: {acc_st*100:.4f}%  meta coefs: {dict(zip(base_names,meta.coef_[0].round(3)))}")
summary_block(f"EXP5: Stacking ({'+'.join(base_names)})",60,"Elo K=48",acc_st,ll_st,br_st,
              f"vs baseline: {(acc_st-BASELINE_ACC)*100:+.3f}pp")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("EXPERIMENT 6 — Best Architecture + G2-Domain Combined")
# ══════════════════════════════════════════════════════════════════════════════
# Identify best architecture from Exp 3-5
arch_pool=[('LR-only',acc3,lr3,'lr'),('Stacking',acc_st,None,'stack')]
arch_pool+= [(nm,ac,m,'single') for nm,m,ac in top2]
best_arch=max(arch_pool,key=lambda x:x[1])
print(f"\n  Best Exp3-5 architecture: {best_arch[0]}  acc={best_arch[1]*100:.4f}%")

# Rebuild Glicko-2 domain features
print(f"[{ts()}] Computing Glicko-2 domain ratings...")
def glicko2_update(mu,phi,sigma,mu_j,phi_j,s,tau=0.5):
    gj=1.0/math.sqrt(1.0+3.0*phi_j**2/math.pi**2)
    E=1.0/(1.0+math.exp(-gj*(mu-mu_j)))
    v=1.0/(gj**2*E*(1.0-E)+1e-10); delta=v*gj*(s-E)
    a=math.log(max(sigma**2,1e-15))
    def f(x):
        ex=math.exp(x); d2=phi**2+v+ex
        return (ex*(delta**2-phi**2-v-ex)/(2.0*d2**2+1e-15))-(x-a)/tau**2
    A=a; B=math.log(max(delta**2-phi**2-v,1e-15)) if delta**2>phi**2+v else a-tau
    fa,fb=f(A),f(B)
    for _ in range(100):
        if abs(B-A)<1e-6: break
        C=A+(A-B)*fa/(fb-fa+1e-15); fc=f(C)
        if fc*fb<0: A,fa=B,fb
        else: fa/=2.0
        B,fb=C,fc
    sn=math.exp(A/2.0); ps=math.sqrt(phi**2+sn**2)
    pn=1.0/math.sqrt(1.0/ps**2+1.0/v)
    return mu+pn**2*gj*(s-E),pn,sn

def compute_g2(df_src,method_kw=None,tau=0.5,SCALE=173.7178,ir=1500,rd=350,si=0.06):
    state={}; rows=[]
    for _,row in df_src.sort_values('date').iterrows():
        r,b=row['R_fighter'],row['B_fighter']
        rm,rp,rs=state.get(r,((ir-1500)/SCALE,rd/SCALE,si))
        bm,bp,bs=state.get(b,((ir-1500)/SCALE,rd/SCALE,si))
        rows+=[{'fighter':r,'date':row['date'],'g2_r':rm*SCALE+1500,'g2_rd':rp*SCALE},
               {'fighter':b,'date':row['date'],'g2_r':bm*SCALE+1500,'g2_rd':bp*SCALE}]
        update=True
        if method_kw:
            m=str(row.get('finish','') or '')
            update=method_kw.lower() in m.lower()
        if update:
            ra=1.0 if row['Winner']=='Red' else(0.0 if row['Winner']=='Blue' else 0.5)
            ba=1.0-ra
            rm_n,rp_n,rs_n=glicko2_update(rm,rp,rs,bm,bp,ra,tau)
            bm_n,bp_n,bs_n=glicko2_update(bm,bp,bs,rm,rp,ba,tau)
            state[r]=(rm_n,rp_n,rs_n); state[b]=(bm_n,bp_n,bs_n)
    return pd.DataFrame(rows)

g2_std=compute_g2(df_all)
g2_ko =compute_g2(df_all,'KO')
g2_sub=compute_g2(df_all,'Sub')
g2_dec=compute_g2(df_all,'Decision')

def merge_g2_col(df_base,g2_df,r_col,b_col,dif_col,fill=1500.0):
    gr=g2_df[['fighter','date','g2_r']].rename(columns={'fighter':'R_fighter','g2_r':r_col})
    gb=g2_df[['fighter','date','g2_r']].rename(columns={'fighter':'B_fighter','g2_r':b_col})
    out=pd.merge_asof(df_base.sort_values('date'),gr.sort_values('date'),
                      on='date',by='R_fighter',direction='backward')
    out=pd.merge_asof(out.sort_values('date'),gb.sort_values('date'),
                      on='date',by='B_fighter',direction='backward')
    out=out.sort_values('date').reset_index(drop=True)
    out[r_col]=out[r_col].fillna(fill); out[b_col]=out[b_col].fillna(fill)
    out[dif_col]=out[r_col]-out[b_col]
    return out

df_g6=df.copy()
df_g6=merge_g2_col(df_g6,g2_std,'R_g2_r','B_g2_r','g2_r_dif')
df_g6=merge_g2_col(df_g6,g2_std.rename(columns={'g2_rd':'g2_r'}) if 'g2_rd' in g2_std.columns
                   else g2_std,'R_g2_r','B_g2_r','g2_r_dif')  # already done
# Actually merge RD separately
gr_rd=g2_std[['fighter','date','g2_rd']].rename(columns={'fighter':'R_fighter','g2_rd':'R_g2_rd'})
gb_rd=g2_std[['fighter','date','g2_rd']].rename(columns={'fighter':'B_fighter','g2_rd':'B_g2_rd'})
df_g6=pd.merge_asof(df_g6.sort_values('date'),gr_rd.sort_values('date'),
                    on='date',by='R_fighter',direction='backward')
df_g6=pd.merge_asof(df_g6.sort_values('date'),gb_rd.sort_values('date'),
                    on='date',by='B_fighter',direction='backward')
df_g6=df_g6.sort_values('date').reset_index(drop=True)
df_g6['R_g2_rd']=df_g6['R_g2_rd'].fillna(350.0); df_g6['B_g2_rd']=df_g6['B_g2_rd'].fillna(350.0)
df_g6['g2_rd_dif']=df_g6['R_g2_rd']-df_g6['B_g2_rd']
df_g6=merge_g2_col(df_g6,g2_ko, 'R_g2_ko','B_g2_ko','g2_ko_dif')
df_g6=merge_g2_col(df_g6,g2_sub,'R_g2_sub','B_g2_sub','g2_sub_dif')
df_g6=merge_g2_col(df_g6,g2_dec,'R_g2_dec','B_g2_dec','g2_dec_dif')
del g2_std,g2_ko,g2_sub,g2_dec; gc.collect()

assert len(df_g6)==len(df), f"G2 merge lost rows: {len(df_g6)} vs {len(df)}"
assert (df_g6['date'].values==df['date'].values).all(), "date mismatch after G2 merge"

ELO_FEATS=['R_elo','B_elo','elo_dif','R_elo_trend','B_elo_trend','elo_trend_dif']
G2_FEATS=['R_g2_r','B_g2_r','g2_r_dif','R_g2_rd','B_g2_rd','g2_rd_dif',
          'g2_ko_dif','g2_sub_dif','g2_dec_dif']
feat_e6=[f for f in best_feats if f not in ELO_FEATS]+G2_FEATS
feat_e6=[f for f in feat_e6 if f in df_g6.columns]
for c in G2_FEATS:
    if c in df_g6.columns:
        df_g6[c]=pd.to_numeric(df_g6[c],errors='coerce').fillna(0.0)

g6_tr=(df_g6['date']<TRAIN_CUTOFF); g6_te=~g6_tr
Xtr_g6_r=df_g6.loc[g6_tr,feat_e6].reset_index(drop=True)
Xte_g6  =df_g6.loc[g6_te,feat_e6].reset_index(drop=True)
ytr_g6_r=df_g6.loc[g6_tr,'target'].reset_index(drop=True)
yte_g6  =df_g6.loc[g6_te,'target'].reset_index(drop=True)
dtr_g6  =df_g6.loc[g6_tr,'date'].reset_index(drop=True)
w_g6r   =pd.Series(compute_weights(dtr_g6),index=ytr_g6_r.index)
Xtr_g6,ytr_g6,wtr_g6=corner_flip(Xtr_g6_r,ytr_g6_r,w_g6r)
print(f"  Exp6 features: {len(feat_e6)}  train={len(Xtr_g6):,}  test={len(Xte_g6):,}")

print(f"[{ts()}] Training Exp6 ({best_arch[0]} + G2-domain)...")
if best_arch[3]=='lr':
    lr_e6=Pipeline([('sc',RobustScaler()),
                    ('lr',LogisticRegression(penalty='l2',C=best_C,solver='liblinear',
                                             max_iter=2000,random_state=SEED))])
    lr_e6.fit(Xtr_g6,ytr_g6,lr__sample_weight=wtr_g6.values)
    p_e6=lr_e6.predict_proba(Xte_g6)
    acc_e6,ll_e6,br_e6=eval_single(p_e6,yte_g6)
elif best_arch[3]=='lr+xgb':
    lr_e6,xgb_e6,plr_e6,px_e6=train_lr_xgb(Xtr_g6,ytr_g6,wtr_g6.values,Xte_g6,yte_g6)
    acc_e6,ll_e6,br_e6,_=eval_blend(plr_e6,px_e6,yte_g6)
else:
    m_e6=clone(best_arch[2])
    try:    m_e6.fit(Xtr_g6,ytr_g6,sample_weight=wtr_g6.values)
    except (TypeError,ValueError): m_e6.fit(Xtr_g6,ytr_g6)
    p_e6=m_e6.predict_proba(Xte_g6)
    acc_e6,ll_e6,br_e6=eval_single(p_e6,yte_g6)

print(f"  Exp6: {acc_e6*100:.4f}%  vs baseline: {(acc_e6-BASELINE_ACC)*100:+.3f}pp")
summary_block(f"EXP6: {best_arch[0]} + G2-Domain (combined)",len(feat_e6),
              "G2-domain",acc_e6,ll_e6,br_e6,
              f"vs baseline: {(acc_e6-BASELINE_ACC)*100:+.3f}pp")
del df_g6; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
print_section("FINAL SUMMARY — ALL EXPERIMENTS RANKED")
# ══════════════════════════════════════════════════════════════════════════════
RS=sorted(RESULTS,key=lambda x:-x['acc'])
print(f"\n  {'Rk':<3}  {'Experiment':<50}  {'N':>5}  {'Rating':<18}  "
      f"{'Acc%':>8}  {'LogLoss':>8}  {'Brier':>7}")
print(f"  {'─'*3}  {'─'*50}  {'─'*5}  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*7}")
for rank,r in enumerate(RS,1):
    is_best =' ◄ BEST'     if rank==1                     else ''
    is_base =' ◄ BASELINE' if 'BASELINE' in r['name']     else ''
    marker  = is_best or is_base
    print(f"  {rank:<3}  {r['name']:<50}  {r['feat_count']:>5}  {r['rating']:<18}  "
          f"{r['acc']*100:>7.4f}%  {r['log_loss']:>8.6f}  {r['brier']:>7.6f}{marker}")

best_r=RS[0]; base_r=next(r for r in RS if 'BASELINE' in r['name'])
delta=(best_r['acc']-base_r['acc'])*100
print(f"\n{'─'*70}")
print(f"  Baseline            : {base_r['acc']*100:.4f}%  (Elo K=48, 129 feats, LR70+XGB30)")
print(f"  Best overall        : {best_r['name']}")
print(f"  Best accuracy       : {best_r['acc']*100:.4f}%")
print(f"  Features / Rating   : {best_r['feat_count']} / {best_r['rating']}")
print(f"  Delta vs baseline   : {delta:+.4f}pp")
if delta>0.0:
    print(f"  ✓  BEATS BASELINE by {delta:.4f}pp")
else:
    print(f"  ✗  Does NOT beat baseline ({delta:.4f}pp)")

# Key takeaways
print(f"\n  Key findings:")
g2_std_r=next((r for r in RS if 'EXP1A' in r['name']),None)
g2_dom_r=next((r for r in RS if 'EXP1B' in r['name']),None)
if g2_std_r: print(f"    • Glicko-2 standard  : {(g2_std_r['acc']-base_r['acc'])*100:+.3f}pp vs Elo")
if g2_dom_r: print(f"    • Glicko-2 domain    : {(g2_dom_r['acc']-base_r['acc'])*100:+.3f}pp vs Elo")
e2_best=max((r for r in RS if r['name'].startswith('EXP2')),key=lambda x:x['acc'])
print(f"    • Best reduced set   : {e2_best['name']} → {e2_best['acc']*100:.4f}%")
e3_r=next((r for r in RS if 'EXP3' in r['name']),None)
if e3_r: print(f"    • LR-only            : {e3_r['acc']*100:.4f}%  ({(e3_r['acc']-base_r['acc'])*100:+.3f}pp)")
st_r=next((r for r in RS if 'EXP5' in r['name']),None)
if st_r: print(f"    • Stacking           : {st_r['acc']*100:.4f}%  ({(st_r['acc']-base_r['acc'])*100:+.3f}pp)")
e6_r=next((r for r in RS if 'EXP6' in r['name']),None)
if e6_r: print(f"    • G2+best arch       : {e6_r['acc']*100:.4f}%  ({(e6_r['acc']-base_r['acc'])*100:+.3f}pp)")
print(f"{'─'*70}")

with open('experiments/research/architecture_sprint_results_v3.json','w') as f:
    json.dump({'baseline_acc':base_r['acc'],'best_acc':best_r['acc'],'delta_pp':delta,
               'best_name':best_r['name'],'results':RS},f,indent=2)
print(f"\n  Results → experiments/research/architecture_sprint_results_v3.json")
print(f"[{ts()}] Sprint complete.\n")
