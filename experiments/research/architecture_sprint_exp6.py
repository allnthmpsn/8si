#!/usr/bin/env python3
"""
architecture_sprint_exp6.py — Exp6 standalone fix.
Best architecture (Stacking: LR + CatBoost-Optuna + RF-Optuna)
+ Glicko-2 domain features replacing Elo.
Appends result to architecture_sprint_results_v3.json.
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
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from catboost import CatBoostClassifier

os.chdir('/Users/allenthompson/Desktop/ufc-predictor')

SEED=42; TRAIN_CUTOFF=pd.Timestamp('2024-01-01'); HL_DAYS=730
CV_FOLDS=5; TRAIN_START='2015-01-01'
WOMENS=["Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight"]

def ts(): return time.strftime('%H:%M:%S')
def compute_weights(dates):
    return np.exp(-np.log(2)*(TRAIN_CUTOFF-dates).dt.days.clip(lower=0)/HL_DAYS)

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

def cv_accuracy(model, X, y, w, skf):
    Xa=X.values if hasattr(X,'values') else X
    ya=y.values if hasattr(y,'values') else y
    wa=w.values if hasattr(w,'values') else w
    scores=[]
    for tri,vai in skf.split(Xa,ya):
        m=clone(model)
        try: m.fit(Xa[tri],ya[tri],sample_weight=wa[tri])
        except (TypeError,ValueError): m.fit(Xa[tri],ya[tri])
        scores.append(accuracy_score(ya[vai],m.predict(Xa[vai])))
    return np.array(scores)

BASELINE_ACC = 0.725000

print(f"[{ts()}] EXP6 — Stacking + G2-Domain")
print("=" * 60)

# ── 1. Rebuild full feature matrix (same as v3 setup) ──────────────
print(f"[{ts()}] Loading data + feature engineering...")
df_all=pd.read_csv('data/ufc-master.csv',low_memory=False)
df_all['date']=pd.to_datetime(df_all['date'])
career_raw=pd.read_csv('data/career_fights_updated.csv')
career_raw['date']=pd.to_datetime(career_raw['date'])
career_raw=career_raw.sort_values(['fighter','date']).reset_index(drop=True)
style_df=pd.read_csv('data/ufc_fighters_final_updated.csv')
for c in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    style_df[c]=pd.to_numeric(style_df[c].astype(str).str.replace('%','',regex=False),
                               errors='coerce').fillna(0.0)/100.0

# Elo history (needed for QA stats)
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
for ra,rd in [('SLpM','SLpM_dif'),('SApM','SApM_dif'),('Str_Def','Str_Def_dif'),
              ('TD_Def','TD_Def_dif'),('Sub_Avg','Sub_Avg_dif'),('TD_Avg','TD_Avg_dif')]:
    df[rd]=df[f'R_{ra}']-df[f'B_{ra}']
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
print(f"  Base df: {len(df):,}  train={(df['date']<TRAIN_CUTOFF).sum()}  test={(df['date']>=TRAIN_CUTOFF).sum()}")

# ── 2. Glicko-2 domain ratings ─────────────────────────────────────
print(f"[{ts()}] Computing Glicko-2 domain ratings (standard + KO + Sub + Decision)...")

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

SCALE=173.7178; IR=1500; RD=350; SI=0.06

def compute_g2_series(df_src, method_kw=None, tau=0.5):
    """Returns DataFrame: fighter, date, g2_r"""
    state={}; rows=[]
    for _,row in df_src.sort_values('date').iterrows():
        r,b=row['R_fighter'],row['B_fighter']
        rm,rp,rs=state.get(r,((IR-1500)/SCALE,RD/SCALE,SI))
        bm,bp,bs=state.get(b,((IR-1500)/SCALE,RD/SCALE,SI))
        rows+=[{'fighter':r,'date':row['date'],'g2_r':rm*SCALE+1500},
               {'fighter':b,'date':row['date'],'g2_r':bm*SCALE+1500}]
        update=True
        if method_kw:
            finish_str=str(row.get('finish','') or '')
            update=method_kw.lower() in finish_str.lower()
        if update:
            ra=1.0 if row['Winner']=='Red' else(0.0 if row['Winner']=='Blue' else 0.5)
            ba=1.0-ra
            rm_n,rp_n,rs_n=glicko2_update(rm,rp,rs,bm,bp,ra,tau)
            bm_n,bp_n,bs_n=glicko2_update(bm,bp,bs,rm,rp,ba,tau)
            state[r]=(rm_n,rp_n,rs_n); state[b]=(bm_n,bp_n,bs_n)
    return pd.DataFrame(rows).sort_values(['fighter','date']).reset_index(drop=True)

def compute_g2_with_rd(df_src, tau=0.5):
    """Standard G2 returning both r and RD."""
    state={}; rows=[]
    for _,row in df_src.sort_values('date').iterrows():
        r,b=row['R_fighter'],row['B_fighter']
        rm,rp,rs=state.get(r,((IR-1500)/SCALE,RD/SCALE,SI))
        bm,bp,bs=state.get(b,((IR-1500)/SCALE,RD/SCALE,SI))
        rows+=[{'fighter':r,'date':row['date'],'g2_r':rm*SCALE+1500,'g2_rd':rp*SCALE},
               {'fighter':b,'date':row['date'],'g2_r':bm*SCALE+1500,'g2_rd':bp*SCALE}]
        ra=1.0 if row['Winner']=='Red' else(0.0 if row['Winner']=='Blue' else 0.5)
        ba=1.0-ra
        rm_n,rp_n,rs_n=glicko2_update(rm,rp,rs,bm,bp,ra,tau)
        bm_n,bp_n,bs_n=glicko2_update(bm,bp,bs,rm,rp,ba,tau)
        state[r]=(rm_n,rp_n,rs_n); state[b]=(bm_n,bp_n,bs_n)
    return pd.DataFrame(rows).sort_values(['fighter','date']).reset_index(drop=True)

g2_full = compute_g2_with_rd(df_all)     # r + RD
g2_ko   = compute_g2_series(df_all,'KO')
g2_sub  = compute_g2_series(df_all,'Sub')
g2_dec  = compute_g2_series(df_all,'Decision')
print(f"  G2 history rows: std={len(g2_full):,}  ko={len(g2_ko):,}  sub={len(g2_sub):,}  dec={len(g2_dec):,}")
gc.collect()

def merge_g2(df_base, g2_df, r_col, b_col, dif_col, val_col='g2_r', fill=1500.0):
    """Safe single-column merge: merge val_col → r_col and b_col."""
    tmp = g2_df[['fighter','date',val_col]].copy()
    gr=tmp.rename(columns={'fighter':'R_fighter', val_col:r_col})
    gb=tmp.rename(columns={'fighter':'B_fighter', val_col:b_col})
    out=pd.merge_asof(df_base.sort_values('date'), gr.sort_values('date'),
                      on='date', by='R_fighter', direction='backward')
    out=pd.merge_asof(out.sort_values('date'), gb.sort_values('date'),
                      on='date', by='B_fighter', direction='backward')
    out=out.sort_values('date').reset_index(drop=True)
    out[r_col]=out[r_col].fillna(fill); out[b_col]=out[b_col].fillna(fill)
    out[dif_col]=out[r_col]-out[b_col]
    return out

df_g6=df.copy()
df_g6=merge_g2(df_g6, g2_full,  'R_g2_r',   'B_g2_r',   'g2_r_dif',   val_col='g2_r')
df_g6=merge_g2(df_g6, g2_full,  'R_g2_rd',  'B_g2_rd',  'g2_rd_dif',  val_col='g2_rd')
df_g6=merge_g2(df_g6, g2_ko,    'R_g2_ko',  'B_g2_ko',  'g2_ko_dif')
df_g6=merge_g2(df_g6, g2_sub,   'R_g2_sub', 'B_g2_sub', 'g2_sub_dif')
df_g6=merge_g2(df_g6, g2_dec,   'R_g2_dec', 'B_g2_dec', 'g2_dec_dif')
del g2_full,g2_ko,g2_sub,g2_dec; gc.collect()

assert len(df_g6)==len(df), f"Merge lost rows: {len(df_g6)} vs {len(df)}"
print(f"  G2 columns merged cleanly — df_g6: {len(df_g6):,} rows")

# ── 3. Build feature set — Top60 with Elo replaced by G2 ──────────
# Load top60 from v3 run's SHAP ranking (reconstructed here from known order)
# We'll use the top60 from SHAP minus elo features, plus G2 features
ELO_FEATS=['R_elo','B_elo','elo_dif','R_elo_trend','B_elo_trend','elo_trend_dif']
G2_NEW=['R_g2_r','B_g2_r','g2_r_dif','R_g2_rd','B_g2_rd','g2_rd_dif',
        'g2_ko_dif','g2_sub_dif','g2_dec_dif']

# Rebuild top60 list using v3 SHAP ranking (same as v3 script — SHAP ran on xgb_base with elo)
# To keep consistent, merge elo back for SHAP then swap; simpler: use all 129 feats minus elo + G2
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
           "SLpM_dif","SApM_dif","Str_Def_dif","TD_Def_dif","Sub_Avg_dif","TD_Avg_dif"]
FEAT_QA=["R_qa_win_rate","R_qa_finish_rate","R_qa_SLpM","R_qa_SApM","B_qa_win_rate",
         "B_qa_finish_rate","B_qa_SLpM","B_qa_SApM","qa_win_rate_dif","qa_finish_rate_dif",
         "qa_SLpM_dif","qa_SApM_dif"]
FEAT_INT=["R_age_x_layoff","B_age_x_layoff","age_x_layoff_dif","R_finish_danger","B_finish_danger",
          "finish_danger_mismatch","R_got_finished_rate","B_got_finished_rate"]
FEAT_ALL_NOELO=[f for f in FEAT_BASE+FEAT_QA+FEAT_INT if f not in ELO_FEATS]

# G2 feature set: top60 proxy = non-elo feats (123 - 6 = 117 feats) + G2 (9) = 126
# We use all non-elo features + G2 (comparable breadth to baseline 129)
feat_e6 = [f for f in FEAT_ALL_NOELO if f in df_g6.columns] + G2_NEW
for c in G2_NEW: df_g6[c]=pd.to_numeric(df_g6[c],errors='coerce').fillna(0.0)
print(f"  Exp6 feature count: {len(feat_e6)}")

train_mask=(df_g6['date']<TRAIN_CUTOFF); test_mask=~train_mask
Xtr_r=df_g6.loc[train_mask,feat_e6].reset_index(drop=True)
ytr_r=df_g6.loc[train_mask,'target'].reset_index(drop=True)
dtr_r=df_g6.loc[train_mask,'date'].reset_index(drop=True)
Xte  =df_g6.loc[test_mask, feat_e6].reset_index(drop=True)
yte  =df_g6.loc[test_mask, 'target'].reset_index(drop=True)
wr=pd.Series(compute_weights(dtr_r),index=ytr_r.index)
Xtr,ytr,wtr=corner_flip(Xtr_r,ytr_r,wr)
warr=wtr.values
print(f"  Train aug: {len(Xtr):,}  Test: {len(Xte):,}")
gc.collect()

# ── 4. Stacking: LR + CatBoost-Optuna + RF-Optuna (same params as Exp5) ─────
print(f"[{ts()}] Rebuilding Exp5 base models on G2 features...")

# Re-tune with brief Optuna on new features (30 trials each) for CatBoost + RF
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
skf=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
N_OPT=30
Xtr_np=Xtr.values; ytr_np=ytr.values; warr_np=wtr.values

def cat_obj(trial):
    m=CatBoostClassifier(
        iterations=trial.suggest_int('it',50,500),
        learning_rate=trial.suggest_float('lr',0.01,0.3,log=True),
        depth=trial.suggest_int('d',2,8),
        l2_leaf_reg=trial.suggest_float('l2',1,10),
        subsample=trial.suggest_float('ss',0.5,1.0),
        random_seed=SEED,verbose=False,thread_count=1)
    return cv_accuracy(m,Xtr,ytr,wtr,skf).mean()

def rf_obj(trial):
    m=RandomForestClassifier(
        n_estimators=trial.suggest_int('n',50,400),
        max_depth=trial.suggest_int('d',3,20),
        min_samples_split=trial.suggest_int('mss',2,20),
        min_samples_leaf=trial.suggest_int('msl',1,10),
        max_features=trial.suggest_float('mf',0.3,1.0),
        random_state=SEED,n_jobs=1)
    return cv_accuracy(m,Xtr,ytr,wtr,skf).mean()

print(f"[{ts()}]  CatBoost Optuna ({N_OPT} trials)...")
scat=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
scat.optimize(cat_obj,n_trials=N_OPT,show_progress_bar=False)
cp=scat.best_params
cat_m=CatBoostClassifier(iterations=cp['it'],learning_rate=cp['lr'],depth=cp['d'],
                          l2_leaf_reg=cp['l2'],subsample=cp['ss'],
                          random_seed=SEED,verbose=False,thread_count=1)
gc.collect()

print(f"[{ts()}]  RF Optuna ({N_OPT} trials)...")
srf=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
srf.optimize(rf_obj,n_trials=N_OPT,show_progress_bar=False)
rfp=srf.best_params
rf_m=RandomForestClassifier(n_estimators=rfp['n'],max_depth=rfp['d'],
                             min_samples_split=rfp['mss'],min_samples_leaf=rfp['msl'],
                             max_features=rfp['mf'],random_state=SEED,n_jobs=1)
gc.collect()

lr_m=Pipeline([('sc',RobustScaler()),
               ('lr',LogisticRegression(penalty='l2',C=10.0,solver='liblinear',
                                        max_iter=2000,random_state=SEED))])
print(f"[{ts()}]  LR best C=10.0")
base_models=[('lr',lr_m),('cat',cat_m),('rf',rf_m)]

# ── 5. OOF stacking ────────────────────────────────────────────────
print(f"[{ts()}] Stacking OOF ({len(base_models)} base models × {CV_FOLDS} folds)...")
Xtr_a=Xtr.values; Xte_a=Xte.values; ytr_a=ytr.values; warr_a=wtr.values
oof=np.zeros((len(Xtr_a),len(base_models)))
tp =np.zeros((len(Xte_a),len(base_models)))

for mi,(nm,m) in enumerate(base_models):
    print(f"  OOF [{mi+1}/{len(base_models)}] {nm}...")
    oc=np.zeros(len(Xtr_a)); tc=np.zeros(len(Xte_a))
    for fold,(tri,vai) in enumerate(skf.split(Xtr_a,ytr_a)):
        mf=clone(m)
        try:    mf.fit(Xtr_a[tri],ytr_a[tri],sample_weight=warr_a[tri])
        except (TypeError,ValueError): mf.fit(Xtr_a[tri],ytr_a[tri])
        oc[vai]=mf.predict_proba(Xtr_a[vai])[:,1]
        tc+=mf.predict_proba(Xte_a)[:,1]/CV_FOLDS
    oof[:,mi]=oc; tp[:,mi]=tc; gc.collect()

meta=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=SEED)
meta.fit(oof,ytr_a)
p_e6=meta.predict_proba(tp)
acc_e6=accuracy_score(yte.values,(p_e6[:,1]>0.5).astype(int))
ll_e6 =log_loss(yte.values,p_e6)
br_e6 =brier_score_loss(yte.values,p_e6[:,1])
delta=acc_e6-BASELINE_ACC

print(f"\n{'='*60}")
print(f"  EXP6 RESULT")
print(f"{'='*60}")
print(f"  Architecture   : Stacking (LR + CatBoost + RF)")
print(f"  Rating system  : Glicko-2 domain (std + KO + Sub + Decision)")
print(f"  Features       : {len(feat_e6)}")
print(f"  Test acc       : {acc_e6*100:.4f}%")
print(f"  Log-loss       : {ll_e6:.6f}")
print(f"  Brier          : {br_e6:.6f}")
print(f"  vs baseline    : {delta*100:+.4f}pp")
print(f"  {'✓ BEATS BASELINE' if delta > 0 else '✗ Does not beat baseline'}")
print(f"  Meta coefs     : {dict(zip([n for n,_ in base_models], meta.coef_[0].round(3)))}")
print(f"{'='*60}")

# ── 6. Final ranked table ───────────────────────────────────────────
RESULTS = [
    dict(name="BASELINE (LR70+XGB30, Elo, 129 feats)",feat_count=129,rating="Elo K=48",
         acc=0.725000,log_loss=0.555988,brier=0.188584,note="Reference"),
    dict(name="EXP1A: Glicko-2 Standard",feat_count=129,rating="G2 tau=0.5",
         acc=0.714583,log_loss=0.558450,brier=0.189734,note="-1.04pp"),
    dict(name="EXP1B: Glicko-2 + Domain",feat_count=132,rating="G2 std+domain",
         acc=0.721875,log_loss=0.558145,brier=0.189579,note="-0.31pp"),
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
    dict(name="EXP3: LR-Only C=10.0, Top60",feat_count=60,rating="Elo K=48",
         acc=0.712500,log_loss=0.556988,brier=0.189882,note="-1.25pp"),
    dict(name="EXP4: LightGBM Default",feat_count=60,rating="Elo K=48",
         acc=0.687500,log_loss=0.585927,brier=0.200396,note="-3.75pp"),
    dict(name="EXP4: LightGBM Optuna",feat_count=60,rating="Elo K=48",
         acc=0.693750,log_loss=0.569000,brier=0.195832,note="-3.13pp"),
    dict(name="EXP4: CatBoost Default",feat_count=60,rating="Elo K=48",
         acc=0.700000,log_loss=0.574177,brier=0.195380,note="-2.50pp"),
    dict(name="EXP4: CatBoost Optuna",feat_count=60,rating="Elo K=48",
         acc=0.709375,log_loss=0.573000,brier=0.196000,note="-1.56pp"),
    dict(name="EXP4: Random Forest Default",feat_count=60,rating="Elo K=48",
         acc=0.703125,log_loss=0.594632,brier=0.203449,note="-2.19pp"),
    dict(name="EXP4: Random Forest Optuna",feat_count=60,rating="Elo K=48",
         acc=0.705208,log_loss=0.586490,brier=0.200133,note="-1.98pp"),
    dict(name="EXP4: SVM RBF (C=1, scale)",feat_count=60,rating="Elo K=48",
         acc=0.718750,log_loss=0.572838,brier=0.194059,note="-0.63pp"),
    dict(name="EXP5: Stacking (LR+CatBoost+RF)",feat_count=60,rating="Elo K=48",
         acc=0.726042,log_loss=0.562019,brier=0.190219,note="+0.10pp ◄ BEATS BASELINE"),
    dict(name="EXP6: Stacking + G2-Domain",feat_count=len(feat_e6),rating="G2 std+domain",
         acc=acc_e6,log_loss=ll_e6,brier=br_e6,
         note=f"{delta*100:+.2f}pp {'BEATS BASELINE' if delta>0 else 'vs baseline'}"),
]

RS=sorted(RESULTS,key=lambda x:-x['acc'])
print(f"\n{'='*72}")
print("  FINAL RANKED TABLE — ALL EXPERIMENTS")
print(f"{'='*72}")
print(f"  {'Rk':<3}  {'Experiment':<48}  {'N':>5}  {'Rating':<16}  {'Acc%':>8}  {'LogLoss':>8}  {'Brier':>7}")
print(f"  {'─'*3}  {'─'*48}  {'─'*5}  {'─'*16}  {'─'*8}  {'─'*8}  {'─'*7}")
for rank,r in enumerate(RS,1):
    is_best=' ◄ BEST' if rank==1 else ''
    is_base=' ◄ BASE' if 'BASELINE' in r['name'] else ''
    marker=is_best or is_base
    print(f"  {rank:<3}  {r['name']:<48}  {r['feat_count']:>5}  {r['rating']:<16}  "
          f"{r['acc']*100:>7.4f}%  {r['log_loss']:>8.6f}  {r['brier']:>7.6f}{marker}")

best_r=RS[0]; base_r=next(r for r in RS if 'BASELINE' in r['name'])
print(f"\n  Baseline : {base_r['acc']*100:.4f}%  (LR70+XGB30, Elo, 129 feats)")
print(f"  Best     : {best_r['name']}")
print(f"           : {best_r['acc']*100:.4f}%  Δ={( best_r['acc']-base_r['acc'])*100:+.4f}pp")
print(f"\n  Key findings:")
print(f"  • All individual alternatives (LR, XGB, LGB, CatBoost, RF, SVM) FAIL to beat baseline")
print(f"  • Stacking (LR + CatBoost + RF) is the only architecture that beats it (+0.10pp)")
print(f"  • Glicko-2 domain features alone do NOT improve on standard Elo (Exp1B: -0.31pp)")
print(f"  • LR (and SVM) both closely match the blended baseline — XGBoost is the drag")
print(f"  • Reducing to Top60 SHAP features loses <0.5pp with LR; XGBoost degrades sharply")
print(f"  • EXP6 result: G2-domain + Stacking = {acc_e6*100:.4f}%  ({delta*100:+.4f}pp vs baseline)")
if delta > 0:
    print(f"  • EXP6 beats baseline — G2-domain + stacking is the recommended new architecture")
else:
    print(f"  • EXP6 does NOT beat baseline — Elo + stacking (Exp5) remains the best result")
print(f"{'─'*72}")

with open('experiments/research/architecture_sprint_results_v3.json','w') as f:
    json.dump({'baseline_acc':base_r['acc'],'best_acc':best_r['acc'],
               'best_name':best_r['name'],'delta_pp':(best_r['acc']-base_r['acc'])*100,
               'results':RS},f,indent=2)
print(f"\n  Saved → experiments/research/architecture_sprint_results_v3.json")
print(f"[{ts()}] EXP6 complete.")
