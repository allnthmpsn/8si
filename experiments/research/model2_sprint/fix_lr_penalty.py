#!/usr/bin/env python3
"""
Task 1 — Retrain M2 LR with penalty='l2', solver='lbfgs'.
Same C (0.292), same 42-feature training set, same StandardScaler.
Compare 50/50 blend accuracy to the elasticnet baseline.
Run from project root: python experiments/research/model2_sprint/fix_lr_penalty.py
"""

import bisect, gc, json, math, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
np.random.seed(42)

SPRINT_DIR   = Path('experiments/research/model2_sprint')
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
LR_WEIGHT_M1 = 0.70
XGB_WEIGHT_M1 = 0.30
SEED = 42

# Confirmed elasticnet C value
C_KEEP = 0.29229052129200933

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
        if odds == 0 or np.isnan(odds): return None
        return abs(odds)/(abs(odds)+100) if odds < 0 else 100/(odds+100)
    except Exception: return None

def novig_probs(f1_odds, f2_odds):
    f1r = implied_prob(f1_odds) or 0.5
    f2r = implied_prob(f2_odds) or 0.5
    t   = f1r + f2r
    return (f1r/t, f2r/t, t-1.0) if t > 0 else (0.5, 0.5, 0.0)

def g(row, col, default=0.0):
    v = row.get(col, default) if isinstance(row, dict) else getattr(row, col, default)
    try:
        if pd.isna(v): return float(default)
    except Exception: pass
    return float(v) if v is not None else float(default)

def layoff_buckets(d):
    return {'lt90':1 if d<90 else 0,'90_180':1 if 90<=d<180 else 0,
            '180_365':1 if 180<=d<365 else 0,'gt365':1 if d>=365 else 0}

print("=" * 60)
print("M2 LR PENALTY FIX: elasticnet → l2")
print(f"C kept at {C_KEEP:.6f}")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────────────────────
model_xgb_m2  = joblib.load('model/ufc_model2_xgb.pkl')          # XGB unchanged
model_lr_m1   = joblib.load('model/ufc_model_best.pkl')
model_xgb_m1  = joblib.load('model/ufc_model_xgb.pkl')
feat_cols_m1  = joblib.load('model/feature_columns_best.pkl')
feat_names_m2 = joblib.load('model/ufc_model2_features.pkl')

with open('model/model2_tier_stats.json') as f:
    tier_stats = json.load(f)
tier_wr_map  = tier_stats['tier_win_rates']
m1_train_acc = tier_stats['m1_train_acc']
m1_wc_acc    = {int(float(k)): v for k, v in tier_stats.get('m1_wc_acc', {}).items()}

print(f"\nM2 feature count: {len(feat_names_m2)}")

# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading data...")
df_master = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_master['date'] = pd.to_datetime(df_master['date'])

career_raw = pd.read_csv('data/career_fights_updated.csv')
career_raw['date'] = pd.to_datetime(career_raw['date'])
career_raw = career_raw.sort_values(['fighter','date']).reset_index(drop=True)

fstats_df = pd.read_csv('data/ufc_fighters_final_updated.csv')
for c in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    fstats_df[c] = pd.to_numeric(fstats_df[c].astype(str).str.replace('%','',regex=False),
                                  errors='coerce').fillna(0)/100.0

elo_hist = pd.read_csv('data/elo_ratings_history.csv')
elo_hist['date'] = pd.to_datetime(elo_hist['date'])
elo_hist = elo_hist.sort_values(['fighter','date']).reset_index(drop=True)

df = df_master[
    (df_master['date'] >= '2018-01-01') &
    df_master['R_odds'].notna() & df_master['B_odds'].notna() &
    df_master['r_dec_odds'].notna() & df_master['b_dec_odds'].notna() &
    df_master['r_sub_odds'].notna() & df_master['b_sub_odds'].notna() &
    df_master['r_ko_odds'].notna() & df_master['b_ko_odds'].notna() &
    df_master['Winner'].isin(['Red','Blue'])
].copy().reset_index(drop=True)

np.random.seed(SEED)
swap_mask = np.random.random(len(df)) < 0.5
r_matched = sorted([c for c in df.columns if c.startswith('R_') and ('B_'+c[2:]) in df.columns])
b_matched = ['B_'+c[2:] for c in r_matched]
for rc, bc in zip(r_matched, b_matched):
    rv=df.loc[swap_mask,rc].values.copy(); bv=df.loc[swap_mask,bc].values.copy()
    df.loc[swap_mask,rc]=bv; df.loc[swap_mask,bc]=rv
df.loc[swap_mask&(df['Winner']=='Red'),'Winner']='TEMP'
df.loc[swap_mask&(df['Winner']=='Blue'),'Winner']='Red'
df.loc[swap_mask&(df['Winner']=='TEMP'),'Winner']='Blue'
for rc,bc in [('r_dec_odds','b_dec_odds'),('r_sub_odds','b_sub_odds'),('r_ko_odds','b_ko_odds')]:
    rv=df.loc[swap_mask,rc].values.copy(); bv=df.loc[swap_mask,bc].values.copy()
    df.loc[swap_mask,rc]=bv; df.loc[swap_mask,bc]=rv

target     = (df['Winner']=='Red').astype(int).values
train_mask = (df['date']<TRAIN_CUTOFF).values
test_mask  = ~train_mask
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]
y_train    = target[train_idx]; y_test = target[test_idx]
print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

# ── Career stats ──────────────────────────────────────────────────────────────
cf = career_raw.copy()
def sc(x): return x.cumsum().shift(1).fillna(0)
cf['cum_fights']      = cf.groupby('fighter').cumcount()
cf['cum_wins']        = cf.groupby('fighter')['won'].transform(sc)
cf['career_win_rate'] = np.where(cf['cum_fights']>0, cf['cum_wins']/cf['cum_fights'], 0.5)
cf['ko_win']  = ((cf['won']==1)&cf['method'].str.contains('KO|TKO',case=False,na=False)).astype(int)
cf['sub_win'] = ((cf['won']==1)&cf['method'].str.contains('Sub|Submission',case=False,na=False)).astype(int)
cf['fin_win'] = ((cf['won']==1)&cf['method'].str.contains('KO|TKO|Sub|Submission',case=False,na=False)).astype(int)
cf['cum_ko']  = cf.groupby('fighter')['ko_win'].transform(sc)
cf['cum_sub'] = cf.groupby('fighter')['sub_win'].transform(sc)
cf['ko_finish_rate']  = np.where(cf['cum_fights']>0, cf['cum_ko']/cf['cum_fights'], 0.0)
cf['sub_finish_rate'] = np.where(cf['cum_fights']>0, cf['cum_sub']/cf['cum_fights'], 0.0)
def rsh(x,n): return x.shift(1).rolling(n,min_periods=1).mean()
cf['last3_win_rate']    = cf.groupby('fighter')['won'].transform(lambda x: rsh(x,3)).fillna(0.5)
cf['last5_won']         = cf.groupby('fighter')['won'].transform(lambda x: rsh(x,5)).fillna(0.5)
cf['last10_win_rate']   = cf.groupby('fighter')['won'].transform(lambda x: rsh(x,10)).fillna(0.5)
cf['last5_finish_rate'] = cf.groupby('fighter')['fin_win'].transform(lambda x: rsh(x,5)).fillna(0.0)
cf['trend_score']       = cf['last3_win_rate']-cf['last10_win_rate']
cf['prev_date']         = cf.groupby('fighter')['date'].shift(1)
cf['layoff_days']       = (cf['date']-cf['prev_date']).dt.days.fillna(365.0)
wr_cache = cf.groupby('fighter')['won'].mean().to_dict()
def oq(grp):
    ops=grp['opponent'].values; res=np.full(len(grp),0.5)
    for i in range(len(grp)):
        prior=ops[max(0,i-5):i]; rates=[wr_cache.get(o,0.5) for o in prior]
        res[i]=float(np.mean(rates)) if rates else 0.5
    return pd.Series(res,index=grp.index)
cf['opp_quality'] = cf.groupby('fighter',group_keys=False).apply(oq)
CAREER_COLS=['cum_fights','career_win_rate','ko_finish_rate','sub_finish_rate',
             'last3_win_rate','last5_won','last10_win_rate','last5_finish_rate',
             'trend_score','layoff_days','opp_quality']
DEFAULT_CAREER={c:(0.5 if 'rate' in c or c=='opp_quality' or c=='trend_score' else
                   (365.0 if c=='layoff_days' else 0)) for c in CAREER_COLS}
cb={}; cd={}
for fn,grp in cf.groupby('fighter'):
    g_=grp.reset_index(drop=True); cb[fn]=g_; cd[fn]=g_['date'].tolist()
def gca(fighter,fdate):
    if fighter not in cb: return DEFAULT_CAREER.copy()
    idx=bisect.bisect_right(cd[fighter],fdate)-1
    if idx<0: return DEFAULT_CAREER.copy()
    return {c:float(cb[fighter].iloc[idx][c]) for c in CAREER_COLS}
eb={}; ed={}
for fn,grp in elo_hist.groupby('fighter'):
    g_=grp.sort_values('date').reset_index(drop=True); eb[fn]=g_; ed[fn]=g_['date'].tolist()
def gea(fighter,fdate):
    if fighter not in eb: return {'elo':1500.0,'elo_trend':0.0}
    idx=bisect.bisect_left(ed[fighter],fdate)-1
    if idx<0: return {'elo':1500.0,'elo_trend':0.0}
    r=eb[fighter].iloc[idx]
    return {'elo':float(r['elo_after']),'elo_trend':float(r.get('elo_trend',0.0) or 0.0)}
fstyle={}
for _,row in fstats_df.iterrows():
    fstyle[row['Fighter_Name']]={k:float(row.get(k,0) or 0) for k in ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']}

# ── 109-feature matrix ────────────────────────────────────────────────────────
print("\nBuilding 109-feature matrix...")
def bfeat(df_row):
    rn=df_row['R_fighter']; bn=df_row['B_fighter']; fd=df_row['date']
    rc=gca(rn,fd); bc=gca(bn,fd); rs=fstyle.get(rn,{}); bs=fstyle.get(bn,{})
    re=gea(rn,fd); be=gea(bn,fd)
    rlb=layoff_buckets(rc['layoff_days']); blb=layoff_buckets(bc['layoff_days'])
    rsp=1 if str(df_row.get('R_Stance','') or '').lower()=='southpaw' else 0
    bsp=1 if str(df_row.get('B_Stance','') or '').lower()=='southpaw' else 0
    rw=g(df_row,'R_wins'); bw=g(df_row,'B_wins'); rl=g(df_row,'R_losses'); bl=g(df_row,'B_losses')
    rh=g(df_row,'R_Height_cms',175); bh=g(df_row,'B_Height_cms',175)
    rrch=g(df_row,'R_Reach_cms',175); brch=g(df_row,'B_Reach_cms',175)
    ra=g(df_row,'R_age',28); ba=g(df_row,'B_age',28)
    rsig=g(df_row,'R_avg_SIG_STR_landed'); bsig=g(df_row,'B_avg_SIG_STR_landed')
    rtd=g(df_row,'R_avg_TD_landed'); btd=g(df_row,'B_avg_TD_landed')
    rws=g(df_row,'R_current_win_streak'); bws=g(df_row,'B_current_win_streak')
    rls=g(df_row,'R_current_lose_streak'); bls=g(df_row,'B_current_lose_streak')
    rlws=g(df_row,'R_longest_win_streak'); blws=g(df_row,'B_longest_win_streak')
    rsp2=g(df_row,'R_avg_SIG_STR_pct'); bsp2=g(df_row,'B_avg_SIG_STR_pct')
    rsub=g(df_row,'R_avg_SUB_ATT'); bsub=g(df_row,'B_avg_SUB_ATT')
    rtdp=g(df_row,'R_avg_TD_pct'); btdp=g(df_row,'B_avg_TD_pct')
    rttb=g(df_row,'R_total_title_bouts'); bttb=g(df_row,'B_total_title_bouts')
    rko=g(df_row,'R_win_by_KO/TKO'); bko=g(df_row,'B_win_by_KO/TKO')
    rsb=g(df_row,'R_win_by_Submission'); bsb=g(df_row,'B_win_by_Submission')
    wc=WC_ORDER.get(str(df_row.get('weight_class','') or ''),6)
    rax=ra*rc['cum_fights']; bax=ba*bc['cum_fights']
    return {
        'R_wins':rw,'R_losses':rl,'R_Height_cms':rh,'R_age':ra,
        'R_avg_SIG_STR_landed':rsig,'R_avg_TD_landed':rtd,
        'R_current_win_streak':rws,'R_current_lose_streak':rls,
        'R_longest_win_streak':rlws,'R_avg_SIG_STR_pct':rsp2,
        'R_avg_SUB_ATT':rsub,'R_avg_TD_pct':rtdp,'R_Reach_cms':rrch,
        'R_total_title_bouts':rttb,'B_wins':bw,'B_losses':bl,
        'B_Height_cms':bh,'B_age':ba,'B_avg_SIG_STR_landed':bsig,
        'B_avg_TD_landed':btd,'B_current_win_streak':bws,'B_current_lose_streak':bls,
        'B_longest_win_streak':blws,'B_avg_SIG_STR_pct':bsp2,'B_avg_SUB_ATT':bsub,
        'B_avg_TD_pct':btdp,'B_Reach_cms':brch,'B_total_title_bouts':bttb,
        'win_dif':rw-bw,'loss_dif':rl-bl,'win_streak_dif':rws-bws,
        'lose_streak_dif':rls-bls,'height_dif':rh-bh,'reach_dif':rrch-brch,
        'age_dif':ra-ba,'sig_str_dif':rsig-bsig,'avg_td_dif':rtd-btd,
        'ko_dif':rko-bko,'sub_dif':rsb-bsb,'total_title_bout_dif':rttb-bttb,
        'weight_class_ord':wc,'orth_clash':1 if (rsp==0 and bsp==0) else 0,
        'south_clash':1 if (rsp==1 and bsp==1) else 0,'R_southpaw':rsp,
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
        'R_age_x_exp':rax,'B_age_x_exp':bax,'age_x_exp_dif':rax-bax,
        'R_layoff_lt90':rlb['lt90'],'R_layoff_90_180':rlb['90_180'],
        'R_layoff_180_365':rlb['180_365'],'R_layoff_gt365':rlb['gt365'],
        'B_layoff_lt90':blb['lt90'],'B_layoff_90_180':blb['90_180'],
        'B_layoff_180_365':blb['180_365'],
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

rows = [bfeat(r) for _,r in df.iterrows()]
X_df = pd.DataFrame(rows, columns=feat_cols_m1)
X_m1 = X_df[feat_cols_m1].values.astype(float)
cm=np.nanmedian(X_m1,axis=0); nm=np.isnan(X_m1); X_m1[nm]=np.take(cm,np.where(nm)[1])
gc.collect()

# ── M1 OOF ────────────────────────────────────────────────────────────────────
print("\nGenerating M1 OOF predictions...")
X_m1_tr=X_m1[train_idx]; X_m1_te=X_m1[test_idx]
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
m1_oof=np.zeros(len(train_idx))
for _,(tri,vai) in enumerate(skf.split(X_m1_tr,y_train)):
    fl=Pipeline([('sc',RobustScaler()),('clf',LogisticRegression(C=0.00711,penalty='l2',max_iter=2000,solver='saga',random_state=SEED))])
    fx=__import__('xgboost').XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=3,subsample=0.8,colsample_bytree=0.8,use_label_encoder=False,eval_metric='logloss',random_state=SEED,verbosity=0,n_jobs=1)
    fl.fit(X_m1_tr[tri],y_train[tri]); fx.fit(X_m1_tr[tri],y_train[tri])
    m1_oof[vai]=LR_WEIGHT_M1*fl.predict_proba(X_m1_tr[vai])[:,1]+XGB_WEIGHT_M1*fx.predict_proba(X_m1_tr[vai])[:,1]
m1_test=LR_WEIGHT_M1*model_lr_m1.predict_proba(X_m1_te)[:,1]+XGB_WEIGHT_M1*model_xgb_m1.predict_proba(X_m1_te)[:,1]
print(f"  M1 test acc: {accuracy_score(y_test,(m1_test>0.5).astype(int)):.4f}")
gc.collect()

# ── 42-feature M2 dataset (identical pipeline to save_candidates.py) ──────────
print("\nBuilding 42-feature M2 dataset...")
BASE_M2=['model1_prob','f1_ml_novig','f2_ml_novig','ml_gap','vig',
         'f1_dec_implied','f2_dec_implied','dec_implied_dif',
         'f1_ko_implied','f2_ko_implied','ko_implied_dif',
         'f1_sub_implied','f2_sub_implied','sub_implied_dif',
         'finish_prob','f1_finish_prob','f2_finish_prob','finish_advantage',
         'abs_gap','vegas_confidence','model_confidence','model_agrees_vegas','gap_x_confidence']
train_pos={v:k for k,v in enumerate(train_idx)}; test_pos={v:k for k,v in enumerate(test_idx)}
m2r=[]
for i,(_,dr) in enumerate(df.iterrows()):
    m1p=float(m1_oof[train_pos[i]]) if i in train_pos else float(m1_test[test_pos[i]])
    f1n,f2n,vig_=novig_probs(dr['R_odds'],dr['B_odds']); ml_gap_=m1p-f1n
    d1=implied_prob(dr['r_dec_odds']) or 0.0; d2=implied_prob(dr['b_dec_odds']) or 0.0
    k1=implied_prob(dr['r_ko_odds']) or 0.0; k2=implied_prob(dr['b_ko_odds']) or 0.0
    s1=implied_prob(dr['r_sub_odds']) or 0.0; s2=implied_prob(dr['b_sub_odds']) or 0.0
    dt=d1+d2; fp=1.0-(dt/2.0) if dt>0 else 0.5; f1f=k1+s1; f2f=k2+s2
    m2r.append([m1p,f1n,f2n,ml_gap_,vig_,d1,d2,d1-d2,k1,k2,k1-k2,s1,s2,s1-s2,
                fp,f1f,f2f,f1f-f2f,abs(ml_gap_),abs(f1n-0.5),abs(m1p-0.5),
                1 if (m1p>0.5)==(f1n>0.5) else 0,ml_gap_*abs(f1n-0.5)])
X2b=np.array(m2r,dtype=float)
cm2=np.nanmedian(X2b,axis=0); nm2=np.isnan(X2b); X2b[nm2]=np.take(cm2,np.where(nm2)[1])

# Step 1
dfh=df[['date','R_fighter','B_fighter','R_odds','B_odds','Winner']].copy()
dfh['f1_won']=(dfh['Winner']=='Red').astype(int); dfh['f1_is_fav']=(dfh['R_odds']<0).astype(int)
ds=dfh.sort_values('date').reset_index()
fb={}; fw={}; db={}; dw={}
s1rows=[None]*len(df)
for _,row in ds.iterrows():
    oi=row['index']; f1=row['R_fighter']; isf=row['f1_is_fav']; won=row['f1_won']
    f1nv,_,_=novig_probs(df.loc[oi,'R_odds'],df.loc[oi,'B_odds'])
    fn=len(fb.get(f1,[])); fw_=fw.get(f1,0); dn=len(db.get(f1,[])); dw_=dw.get(f1,0)
    t=0 if f1nv<0.30 else(1 if f1nv<0.45 else(2 if f1nv<0.55 else(3 if f1nv<0.70 else 4)))
    s1rows[oi]=[isf,fw_/fn if fn>0 else 0.5,dw_/dn if dn>0 else 0.5,
                math.log1p(fn),math.log1p(dn),abs(f1nv-0.5),t]
    if isf: fb.setdefault(f1,[]).append(row['date']); fw[f1]=fw.get(f1,0)+won
    else:   db.setdefault(f1,[]).append(row['date']); dw[f1]=dw.get(f1,0)+won
s1a=np.array(s1rows,dtype=float)
thr_wr=np.array([float(tier_wr_map.get(str(int(t)),0.5)) for t in s1a[:,6]])
s1f=np.column_stack([s1a[:,:6],thr_wr])

# Step 2
s2r=[]
for i in range(len(df)):
    br=X2b[i]
    k1i=br[BASE_M2.index('f1_ko_implied')]; k2i=br[BASE_M2.index('f2_ko_implied')]
    s1i=br[BASE_M2.index('f1_sub_implied')]; s2i=br[BASE_M2.index('f2_sub_implied')]
    d1i=br[BASE_M2.index('f1_dec_implied')]; d2i=br[BASE_M2.index('f2_dec_implied')]
    m1pi=br[BASE_M2.index('model1_prob')]; fpi=br[BASE_M2.index('finish_prob')]
    rko=X_m1[i,feat_cols_m1.index('R_ko_finish_rate')]; bko=X_m1[i,feat_cols_m1.index('B_ko_finish_rate')]
    rsb=X_m1[i,feat_cols_m1.index('R_sub_finish_rate')]; bsb=X_m1[i,feat_cols_m1.index('B_sub_finish_rate')]
    sdd=X_m1[i,feat_cols_m1.index('Str_Def_dif')]
    s2r.append([k1i*rko-k2i*bko,s1i*rsb-s2i*bsb,fpi*abs(m1pi-0.5),
                ((d1i+d2i)/2.0)*abs(sdd),k1i+k2i,s1i+s2i,abs(k1i-k2i),abs(s1i-s2i)])
s2a=np.array(s2r,dtype=float)

# Step 3
wca=X_m1[:,feat_cols_m1.index('weight_class_ord')]
nrd=df.get('no_of_rounds',pd.Series([3]*len(df))).fillna(3).values
s3r=[]
for i in range(len(df)):
    wv=wca[i]; is5=1 if nrd[i]>=5 else 0
    wa=m1_wc_acc.get(int(wv),m1_train_acc); mc=abs(X2b[i,BASE_M2.index('model_confidence')])
    s3r.append([wv/11.0,is5,wa-m1_train_acc,is5*mc])
s3a=np.array(s3r,dtype=float)

X2_full=np.hstack([X2b,s1f,s2a,s3a])
X2_tr=X2_full[train_idx]; X2_te=X2_full[test_idx]
print(f"  Full M2 matrix: {X2_full.shape}")
gc.collect()

# ── Baseline: ElasticNet LR + XGB 50/50 ──────────────────────────────────────
elas_lr  = joblib.load('model/ufc_model2_best.pkl')   # current elasticnet
xgb_m2   = model_xgb_m2

df_te = pd.DataFrame(X2_te, columns=feat_names_m2)
elas_prob  = elas_lr.predict_proba(df_te)[:,1]
xgb_prob   = xgb_m2.predict_proba(df_te)[:,1]
blend_elas = 0.50*elas_prob + 0.50*xgb_prob
acc_elas   = accuracy_score(y_test,(blend_elas>0.5).astype(int))
print(f"\nBaseline (ElasticNet LR 50/50 blend) test acc: {acc_elas:.4f}")

# ── Retrain with L2, lbfgs, same C ───────────────────────────────────────────
print(f"\nRetraining LR: penalty=l2, solver=lbfgs, C={C_KEEP:.6f}, class_weight=balanced...")

new_lr = Pipeline([
    ('sc',  StandardScaler()),
    ('clf', LogisticRegression(
        C=C_KEEP,
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced',
        max_iter=2000,
        random_state=SEED,
    ))
])
new_lr.fit(X2_tr, y_train)

df_tr_in = pd.DataFrame(X2_tr, columns=feat_names_m2)
lr_train_acc = accuracy_score(y_train, new_lr.predict(df_tr_in))

new_lr_prob  = new_lr.predict_proba(df_te)[:,1]
blend_l2     = 0.50*new_lr_prob + 0.50*xgb_prob
acc_l2       = accuracy_score(y_test,(blend_l2>0.5).astype(int))
acc_lr_only  = accuracy_score(y_test,(new_lr_prob>0.5).astype(int))

print(f"  L2 LR train acc      : {lr_train_acc:.4f}")
print(f"  L2 LR test acc       : {acc_lr_only:.4f}")
print(f"  L2 LR 50/50 blend    : {acc_l2:.4f}")
print(f"  Elasticnet 50/50     : {acc_elas:.4f}")
delta = acc_l2 - acc_elas
print(f"  Delta (L2 - Elas)    : {delta:+.4f} ({'+' if delta>0 else ''}{delta*100:.2f}pp)")

# Confirm L2 params
print(f"\n  Confirmed params:")
for k,v in new_lr.get_params().items():
    if k in ('clf__penalty','clf__solver','clf__C','clf__class_weight','clf__max_iter'):
        print(f"    {k}: {v}")

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving updated LR model to model/ufc_model2_best.pkl...")
joblib.dump(new_lr, 'model/ufc_model2_best.pkl')

# Also update sprint candidate for consistency
joblib.dump(new_lr, SPRINT_DIR / 'model2_candidate_lr.pkl')

# Update model_metadata.json
with open('model/model_metadata.json') as f:
    meta = json.load(f)

meta['model2']['temporal_accuracy'] = round(acc_l2, 4)
meta['model2']['lr_penalty']        = 'l2'
meta['model2']['lr_solver']         = 'lbfgs'
meta['model2']['lr_C']              = round(C_KEEP, 6)
meta['model2']['blend_ratio']       = '50% LR (l2) + 50% XGB'
meta['model2']['note_penalty_fix']  = f'Retrained from elasticnet to l2; delta={delta:+.4f}'

with open('model/model_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"  model_metadata.json updated")

# Verify reload
verify = joblib.load('model/ufc_model2_best.pkl')
vp = verify.get_params()
print(f"\nVerification — reloaded model:")
print(f"  penalty : {vp['clf__penalty']}")
print(f"  solver  : {vp['clf__solver']}")
print(f"  C       : {vp['clf__C']:.6f}")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"  ElasticNet 50/50 blend : {acc_elas:.4f}")
print(f"  L2 LR      50/50 blend : {acc_l2:.4f}  ({delta:+.4f} / {delta*100:+.2f}pp)")
print(f"  Model saved to model/ufc_model2_best.pkl")
print(f"{'='*60}")
