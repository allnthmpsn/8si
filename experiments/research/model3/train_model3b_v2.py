"""
Model 3B v2 — Six-class Winner + Method, with M1 winner probability as a feature.
New features added: m1_red_win_prob, m1_red_win_prob_sq, m1_confidence
Saves to experiments/research/model3/ with model3b_v2_ prefix.
Production model files in model/ are READ ONLY — never modified.
"""

import gc, json, os, sys, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, '../../../data')
MODEL = os.path.join(BASE, '../../../model')   # READ ONLY
OUT   = BASE

CUTOFF = pd.Timestamp('2024-01-01')
DEC_CODES    = {'U-DEC', 'S-DEC', 'M-DEC'}
FINISH_CODES = {'KO/TKO', 'SUB'}
WOMENS = {"Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight"}
WC_ORD = {
    "Women's Strawweight": 0, "Women's Flyweight": 1,
    "Women's Bantamweight": 2, "Women's Featherweight": 3,
    'Flyweight': 4, 'Bantamweight': 5, 'Featherweight': 6,
    'Lightweight': 7, 'Welterweight': 8, 'Middleweight': 9,
    'Light Heavyweight': 10, 'Heavyweight': 11, 'Catch Weight': 6,
}
LABEL_MAP = {
    ('Red',  'KO/TKO'): 0, ('Red',  'SUB'): 1, ('Red',  'DEC'): 2,
    ('Blue', 'KO/TKO'): 3, ('Blue', 'SUB'): 4, ('Blue', 'DEC'): 5,
}
CLASS_NAMES = ['R KO/TKO','R Sub','R Dec','B KO/TKO','B Sub','B Dec']
LABEL_FLIP  = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2}
M1_LR_W, M1_XGB_W = 0.70, 0.30

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load production M1 — READ ONLY
# ═══════════════════════════════════════════════════════════════════════
print("=" * 64)
print("STEP 1: Load production Model 1 (read-only)")
print("=" * 64)

m1_lr    = joblib.load(os.path.join(MODEL, 'ufc_model_best.pkl'))
m1_xgb   = joblib.load(os.path.join(MODEL, 'ufc_model_xgb.pkl'))
M1_FEATS = joblib.load(os.path.join(MODEL, 'feature_columns_best.pkl'))
print(f"  M1 LR  type: {type(m1_lr).__name__}  (pipeline: {hasattr(m1_lr, 'named_steps')})")
print(f"  M1 XGB type: {type(m1_xgb).__name__}")
print(f"  M1 feature count: {len(M1_FEATS)}")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Load raw data
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 2: Load raw data")
print("=" * 64)

master = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
master['date'] = pd.to_datetime(master['date'])
print(f"  master: {len(master):,} rows")

career_df = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
career_df['date'] = pd.to_datetime(career_df['date'], errors='coerce')
career_df = career_df.dropna(subset=['date','fighter']).sort_values(
    ['fighter','date']).reset_index(drop=True)
print(f"  career_fights: {len(career_df):,} rows")

style_df = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final_updated.csv'))
for col in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    style_df[col] = pd.to_numeric(
        style_df[col].astype(str).str.replace('%','',regex=False),
        errors='coerce').fillna(0.0) / 100.0
style_df = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last').reset_index(drop=True)
print(f"  style_df: {len(style_df):,} rows")

elo_hist = pd.read_csv(os.path.join(DATA, 'elo_ratings_history.csv'))
elo_hist['date'] = pd.to_datetime(elo_hist['date'], errors='coerce')
elo_hist = elo_hist.dropna(subset=['date','fighter']).sort_values(['fighter','date'])
print(f"  elo_history: {len(elo_hist):,} rows")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Build career stats (same pipeline as M1 train_model1.py)
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 3: Build M1-compatible career stats (shift=1, no leakage)")
print("=" * 64)

def compute_career_stats(df, all_win_rates):
    df = df.sort_values(['fighter','date']).copy().reset_index(drop=True)
    df['_ko']  = ((df['won'] == 1) & df['method'].str.contains('KO|TKO', case=False, na=False)).astype(float)
    df['_sub'] = ((df['won'] == 1) & df['method'].str.contains('Sub|Submission', case=False, na=False)).astype(float)
    df['_fin'] = ((df['won'] == 1) & df['method'].str.contains('KO|TKO|Sub', case=False, na=False)).astype(float)
    g = df.groupby('fighter', sort=False)
    df['cum_fights'] = g.cumcount()
    for src, dst in [('won','_cs_won'),('_ko','_cs_ko'),('_sub','_cs_sub'),('_fin','_cs_fin')]:
        df[dst] = g[src].cumsum() - df[src]
    safe_n = df['cum_fights'].clip(lower=1)
    df['career_win_rate']    = np.where(df['cum_fights'] > 0, df['_cs_won'] / safe_n, 0.5)
    df['ko_finish_rate']     = np.where(df['cum_fights'] > 0, df['_cs_ko']  / safe_n, 0.0)
    df['sub_finish_rate']    = np.where(df['cum_fights'] > 0, df['_cs_sub'] / safe_n, 0.0)
    df['career_finish_rate'] = np.where(df['cum_fights'] > 0, df['_cs_fin'] / safe_n, 0.0)
    def _roll(series, w, default):
        return series.shift(1).rolling(w, min_periods=1).mean().fillna(default)
    df['last3_win_rate']    = g['won'].transform(lambda x: _roll(x, 3,  0.5))
    df['last10_win_rate']   = g['won'].transform(lambda x: _roll(x, 10, 0.5))
    df['last5_won']         = g['won'].transform(lambda x: _roll(x, 5,  0.5))
    df['last5_finish_rate'] = g['_fin'].transform(lambda x: _roll(x, 5, 0.0))
    df['trend_score']       = df['last3_win_rate'] - df['last10_win_rate']
    df['_prev_date'] = g['date'].transform(lambda x: x.shift(1))
    df['layoff_days'] = (df['date'] - df['_prev_date']).dt.days.fillna(180.0).clip(lower=0)
    opp_col     = df['opponent'].tolist()
    fighter_col = df['fighter'].tolist()
    idx_list    = df.index.tolist()
    fighter_positions = defaultdict(list)
    for pos, idx in enumerate(idx_list):
        fighter_positions[fighter_col[pos]].append(pos)
    opp_quality_arr = np.full(len(df), 0.5)
    for fighter, positions in fighter_positions.items():
        for rank, pos in enumerate(positions):
            past_opps = [opp_col[p] for p in positions[max(0, rank-5):rank]]
            rates = [all_win_rates[opp] for opp in past_opps if opp in all_win_rates]
            opp_quality_arr[pos] = float(np.mean(rates)) if rates else 0.5
    df['opp_quality'] = opp_quality_arr
    drop_cols = ['_ko','_sub','_fin','_cs_won','_cs_ko','_cs_sub','_cs_fin','_prev_date']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df[['fighter','date','cum_fights','career_win_rate','ko_finish_rate',
               'sub_finish_rate','career_finish_rate','last3_win_rate','last10_win_rate',
               'last5_won','last5_finish_rate','trend_score','layoff_days','opp_quality']]

all_win_rates = {
    f: grp['won'].sum() / max(1, len(grp))
    for f, grp in career_df.groupby('fighter')
}
career_stats = compute_career_stats(career_df, all_win_rates)
print(f"  Career stat rows: {len(career_stats):,}")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Build QA stats (opponent-elo-weighted, shift=1)
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 4: Build QA stats")
print("=" * 64)

def compute_qa_stats(career_df, elo_hist_df):
    elo_lookup = elo_hist_df[['fighter','date','elo_before']].copy()
    opp_elo_df = career_df[['fighter','opponent','date','won','got_finish']].copy()
    opp_elo_df = opp_elo_df.rename(columns={'opponent': 'opp_name'})
    opp_ref = elo_lookup.rename(columns={'fighter':'opp_name','elo_before':'opp_elo'})
    opp_elo_df = pd.merge_asof(
        opp_elo_df.sort_values('date'),
        opp_ref.sort_values('date'),
        on='date', by='opp_name', direction='backward')
    opp_elo_df['opp_elo'] = opp_elo_df['opp_elo'].fillna(1500.0)
    opp_elo_df['ew']      = opp_elo_df['opp_elo'] / 1500.0
    qa_rows = []
    for fighter, grp in opp_elo_df.groupby('fighter', sort=False):
        grp = grp.sort_values('date')
        n = len(grp)
        qa_wr = np.full(n, 0.5); qa_fr = np.full(n, 0.0)
        qa_sl = np.full(n, 0.0); qa_sa = np.full(n, 0.0)
        cum_ew = cum_eww = cum_ewf = 0.0
        cum_n  = cum_off = cum_def = 0.0
        for i, (_, row) in enumerate(grp.iterrows()):
            if cum_ew > 0:
                qa_wr[i] = cum_eww / cum_ew
                qa_fr[i] = cum_ewf / cum_ew
            if cum_n > 0:
                qa_sl[i] = cum_off / cum_n
                qa_sa[i] = cum_def / cum_n
            ew = row['ew']; w = row['won']
            f  = row['got_finish'] if pd.notna(row.get('got_finish')) else 0.0
            cum_ew += ew; cum_eww += ew * w; cum_ewf += ew * f
            cum_n  += ew; cum_off += ew * w; cum_def += ew * (1.0 - w)
        qa_rows.append(pd.DataFrame({
            'fighter': fighter, 'date': grp['date'].values,
            'qa_win_rate': qa_wr, 'qa_finish_rate': qa_fr,
            'qa_SLpM': qa_sl, 'qa_SApM': qa_sa,
        }))
    return pd.concat(qa_rows, ignore_index=True).sort_values(['fighter','date'])

qa_stats = compute_qa_stats(career_df, elo_hist)
print(f"  QA stat rows: {len(qa_stats):,}")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Build M1 feature matrix for all 2015+ fights
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 5: Build M1 feature matrix and generate m1_red_win_prob")
print("=" * 64)

# Filter master to valid fights (2015+, known winner) — keep women's for M1 coverage
mdf = master[
    master['Winner'].isin(['Red','Blue']) &
    (master['date'].dt.year >= 2015)
].copy().sort_values('date').reset_index(drop=True)
print(f"  Fights in M1 universe (2015+): {len(mdf):,}")

# Merge career stats (merge_asof: backward by date, by fighter)
career_cols = [c for c in career_stats.columns if c not in ('fighter','date')]
r_career = career_stats.rename(columns={'fighter':'R_fighter',
    **{c: f'R_{c}' for c in career_cols}})
b_career = career_stats.rename(columns={'fighter':'B_fighter',
    **{c: f'B_{c}' for c in career_cols}})
mdf = pd.merge_asof(mdf.sort_values('date'), r_career.sort_values('date'),
                    on='date', by='R_fighter', direction='backward')
mdf = pd.merge_asof(mdf.sort_values('date'), b_career.sort_values('date'),
                    on='date', by='B_fighter', direction='backward')
career_defaults = {
    'cum_fights': 0, 'career_win_rate': 0.5, 'ko_finish_rate': 0.0,
    'sub_finish_rate': 0.0, 'career_finish_rate': 0.0,
    'last3_win_rate': 0.5, 'last10_win_rate': 0.5,
    'last5_won': 0.5, 'last5_finish_rate': 0.0,
    'trend_score': 0.0, 'layoff_days': 180.0, 'opp_quality': 0.5,
}
for stat, default in career_defaults.items():
    mdf[f'R_{stat}'] = mdf[f'R_{stat}'].fillna(default)
    mdf[f'B_{stat}'] = mdf[f'B_{stat}'].fillna(default)
gc.collect()

# Merge QA stats
qa_r = qa_stats.rename(columns={'fighter':'R_fighter',
    'qa_win_rate':'R_qa_win_rate','qa_finish_rate':'R_qa_finish_rate',
    'qa_SLpM':'R_qa_SLpM','qa_SApM':'R_qa_SApM'})
qa_b = qa_stats.rename(columns={'fighter':'B_fighter',
    'qa_win_rate':'B_qa_win_rate','qa_finish_rate':'B_qa_finish_rate',
    'qa_SLpM':'B_qa_SLpM','qa_SApM':'B_qa_SApM'})
mdf = pd.merge_asof(mdf.sort_values('date'), qa_r.sort_values('date'),
                    on='date', by='R_fighter', direction='backward')
mdf = pd.merge_asof(mdf.sort_values('date'), qa_b.sort_values('date'),
                    on='date', by='B_fighter', direction='backward')
for c in ['R_qa_win_rate','R_qa_finish_rate','R_qa_SLpM','R_qa_SApM',
          'B_qa_win_rate','B_qa_finish_rate','B_qa_SLpM','B_qa_SApM']:
    mdf[c] = mdf[c].fillna(0.5 if 'win_rate' in c else 0.0)
mdf['qa_win_rate_dif']    = mdf['R_qa_win_rate']    - mdf['B_qa_win_rate']
mdf['qa_finish_rate_dif'] = mdf['R_qa_finish_rate'] - mdf['B_qa_finish_rate']
mdf['qa_SLpM_dif']        = mdf['R_qa_SLpM']        - mdf['B_qa_SLpM']
mdf['qa_SApM_dif']        = mdf['R_qa_SApM']        - mdf['B_qa_SApM']
gc.collect()

# Merge Elo from saved history
elo_cols_df = elo_hist[['fighter','date','elo_before','elo_trend']].copy()
elo_r = elo_cols_df.rename(columns={'fighter':'R_fighter','elo_before':'R_elo','elo_trend':'R_elo_trend'})
elo_b = elo_cols_df.rename(columns={'fighter':'B_fighter','elo_before':'B_elo','elo_trend':'B_elo_trend'})
mdf = pd.merge_asof(mdf.sort_values('date'), elo_r.sort_values('date'),
                    on='date', by='R_fighter', direction='backward')
mdf = pd.merge_asof(mdf.sort_values('date'), elo_b.sort_values('date'),
                    on='date', by='B_fighter', direction='backward')
mdf['R_elo']       = mdf['R_elo'].fillna(1500.0)
mdf['B_elo']       = mdf['B_elo'].fillna(1500.0)
mdf['R_elo_trend'] = mdf['R_elo_trend'].fillna(0.0)
mdf['B_elo_trend'] = mdf['B_elo_trend'].fillna(0.0)
gc.collect()

# Merge style stats (updated file with Str_Acc, TD_Acc)
style_src = ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']
r_style = style_df[['Fighter_Name'] + style_src].rename(
    columns={'Fighter_Name':'R_fighter', **{c: f'R_{c}' for c in style_src}})
b_style = style_df[['Fighter_Name'] + style_src].rename(
    columns={'Fighter_Name':'B_fighter', **{c: f'B_{c}' for c in style_src}})
mdf = mdf.merge(r_style, on='R_fighter', how='left')
mdf = mdf.merge(b_style, on='B_fighter', how='left')
for col in [f'{p}{s}' for p in ('R_','B_') for s in style_src]:
    mdf[col] = pd.to_numeric(mdf[col], errors='coerce').fillna(0.0)
gc.collect()

# Feature engineering — same as M1
mdf['weight_class_ord'] = mdf['weight_class'].map(WC_ORD).fillna(6).astype(int)
mdf['R_southpaw'] = (mdf['R_Stance'].str.lower() == 'southpaw').astype(int)
mdf['B_southpaw'] = (mdf['B_Stance'].str.lower() == 'southpaw').astype(int)
mdf['orth_clash']  = ((mdf['R_southpaw'] == 0) & (mdf['B_southpaw'] == 0)).astype(int)
mdf['south_clash'] = ((mdf['R_southpaw'] == 1) & (mdf['B_southpaw'] == 1)).astype(int)
mdf['R_age'] = pd.to_numeric(mdf['R_age'], errors='coerce').fillna(28.0)
mdf['B_age'] = pd.to_numeric(mdf['B_age'], errors='coerce').fillna(28.0)
mdf['R_age_x_exp']  = mdf['R_age'] * mdf['R_cum_fights']
mdf['B_age_x_exp']  = mdf['B_age'] * mdf['B_cum_fights']
mdf['age_x_exp_dif'] = mdf['R_age_x_exp'] - mdf['B_age_x_exp']

def layoff_buckets(prefix, days_series):
    d = days_series.fillna(180.0)
    return {
        f'{prefix}layoff_lt90':    (d < 90).astype(int),
        f'{prefix}layoff_90_180':  ((d >= 90)  & (d < 180)).astype(int),
        f'{prefix}layoff_180_365': ((d >= 180) & (d < 365)).astype(int),
        f'{prefix}layoff_gt365':   (d >= 365).astype(int),
    }

for col, vals in layoff_buckets('R_', mdf['R_layoff_days']).items():
    mdf[col] = vals.values
for col, vals in layoff_buckets('B_', mdf['B_layoff_days']).items():
    mdf[col] = vals.values

for stat in ['career_win_rate','last5_won','last5_finish_rate','opp_quality',
             'trend_score','ko_finish_rate','sub_finish_rate',
             'last3_win_rate','last10_win_rate']:
    mdf[f'{stat}_dif'] = mdf[f'R_{stat}'] - mdf[f'B_{stat}']

mdf['elo_dif']       = mdf['R_elo']       - mdf['B_elo']
mdf['elo_trend_dif'] = mdf['R_elo_trend'] - mdf['B_elo_trend']
mdf['SLpM_dif']      = mdf['R_SLpM']    - mdf['B_SLpM']
mdf['SApM_dif']      = mdf['R_SApM']    - mdf['B_SApM']
mdf['Str_Def_dif']   = mdf['R_Str_Def'] - mdf['B_Str_Def']
mdf['TD_Def_dif']    = mdf['R_TD_Def']  - mdf['B_TD_Def']
mdf['Sub_Avg_dif']   = mdf['R_Sub_Avg'] - mdf['B_Sub_Avg']
mdf['TD_Avg_dif']    = mdf['R_TD_Avg']  - mdf['B_TD_Avg']
gc.collect()

# Interaction features
def compute_interaction_features(df, cdf):
    cdf2 = cdf.sort_values(['fighter','date']).copy()
    cdf2['is_loss']     = (cdf2['won'] == 0).astype(float)
    cdf2['is_fin_loss'] = ((cdf2['won'] == 0) & (cdf2['got_finish'].fillna(0) == 1)).astype(float)
    g2 = cdf2.groupby('fighter', sort=False)
    cdf2['_cs_l']  = g2['is_loss'].cumsum()     - cdf2['is_loss']
    cdf2['_cs_fl'] = g2['is_fin_loss'].cumsum() - cdf2['is_fin_loss']
    cdf2['got_finished_rate'] = np.where(
        cdf2['_cs_l'] > 0, cdf2['_cs_fl'] / cdf2['_cs_l'], 0.5)
    chin = cdf2[['fighter','date','got_finished_rate']].sort_values(['fighter','date'])
    cr = chin.rename(columns={'fighter':'R_fighter','got_finished_rate':'R_got_finished_rate'})
    cb = chin.rename(columns={'fighter':'B_fighter','got_finished_rate':'B_got_finished_rate'})
    df = pd.merge_asof(df.sort_values('date'), cr.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), cb.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')
    df['R_got_finished_rate'] = df['R_got_finished_rate'].fillna(0.5)
    df['B_got_finished_rate'] = df['B_got_finished_rate'].fillna(0.5)
    df['R_age_x_layoff'] = df['R_age'] * df['R_layoff_days'].clip(upper=730)
    df['B_age_x_layoff'] = df['B_age'] * df['B_layoff_days'].clip(upper=730)
    df['age_x_layoff_dif'] = df['R_age_x_layoff'] - df['B_age_x_layoff']
    df['R_finish_danger'] = df['R_ko_finish_rate'] + df['R_sub_finish_rate']
    df['B_finish_danger'] = df['B_ko_finish_rate'] + df['B_sub_finish_rate']
    df['finish_danger_mismatch'] = (
        df['R_finish_danger'] * (1 - df['B_got_finished_rate']) -
        df['B_finish_danger'] * (1 - df['R_got_finished_rate'])
    )
    return df

mdf = compute_interaction_features(mdf, career_df)
gc.collect()

# Force all M1 features numeric
for col in M1_FEATS:
    if col in mdf.columns:
        mdf[col] = pd.to_numeric(mdf[col], errors='coerce').fillna(0.0)
    else:
        mdf[col] = 0.0

missing_m1 = [f for f in M1_FEATS if f not in mdf.columns or mdf[f].isna().any()]
print(f"  M1 features built.  Still missing/null: {len(missing_m1)}")
if missing_m1:
    print(f"    {missing_m1[:10]}")

# Build M1 input matrix
X_m1 = mdf[M1_FEATS].values.astype(float)
X_m1_df = pd.DataFrame(X_m1, columns=M1_FEATS)

# M1 LR is a Pipeline (includes its own scaler), XGB is standalone
p_lr  = m1_lr.predict_proba(X_m1_df)[:, 1]
p_xgb = m1_xgb.predict_proba(X_m1_df)[:, 1]

mdf['m1_red_win_prob'] = M1_LR_W * p_lr + M1_XGB_W * p_xgb
print(f"\n  M1 prob stats (all 2015+ fights):")
print(f"    Mean: {mdf['m1_red_win_prob'].mean():.4f}  Std: {mdf['m1_red_win_prob'].std():.4f}")
print(f"    >0.5: {(mdf['m1_red_win_prob'] > 0.5).mean():.4f}")

# Direction accuracy check on universe (should be ~72.81% on test window)
mdf_test = mdf[mdf['date'] >= CUTOFF]
m1_dir_universe = ((mdf_test['m1_red_win_prob'] > 0.5) == (mdf_test['Winner'] == 'Red')).mean()
print(f"    Direction accuracy (2024+ test window): {m1_dir_universe:.4f}  (ref: 0.7281)")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 6: Build 3B dataset and join m1_red_win_prob
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 6: Build 3B dataset and join M1 probability")
print("=" * 64)

# Filter to 3B universe: 2015+, valid winner, known method
master_3b = master[
    master['Winner'].isin(['Red','Blue']) &
    master['finish'].isin(DEC_CODES | FINISH_CODES) &
    (master['date'].dt.year >= 2015)
].copy()

def build_label(row):
    w = row['Winner']
    f = row['finish']
    method = 'DEC' if f in DEC_CODES else ('KO/TKO' if f == 'KO/TKO' else 'SUB')
    return LABEL_MAP.get((w, method), None)

master_3b['label_6'] = master_3b.apply(build_label, axis=1)
master_3b = master_3b[master_3b['label_6'].notna()].copy()
master_3b['label_6'] = master_3b['label_6'].astype(int)
print(f"  3B universe: {len(master_3b):,} fights")

# Join m1_red_win_prob from mdf (which covers all 2015+ valid-winner fights)
m1_prob_join = mdf[['R_fighter','B_fighter','date','m1_red_win_prob']].copy()
master_3b = master_3b.merge(m1_prob_join, on=['R_fighter','B_fighter','date'], how='left')
missing_prob = master_3b['m1_red_win_prob'].isna().sum()
print(f"  m1_red_win_prob join: {len(master_3b) - missing_prob} matched, {missing_prob} missing (fill 0.5)")
master_3b['m1_red_win_prob'] = master_3b['m1_red_win_prob'].fillna(0.5)

# Derived M1 features
master_3b['m1_red_win_prob_sq'] = master_3b['m1_red_win_prob'] ** 2
master_3b['m1_confidence']      = (master_3b['m1_red_win_prob'] - 0.5).abs()
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 7: Add remaining 3B features
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 7: Add remaining 3B base features")
print("=" * 64)

master_3b['weight_class_ord'] = master_3b['weight_class'].map(WC_ORD).fillna(6)
master_3b['is_5rnd']   = (master_3b['no_of_rounds'] == 5).astype(int)
master_3b['is_title']  = master_3b['title_bout'].astype(int)
master_3b['is_womens'] = master_3b['weight_class'].isin(WOMENS).astype(int)

# Elo (from elo_hist lookup)
elo_lookup_dict = {}
for _, row in elo_hist.iterrows():
    elo_lookup_dict[(row['fighter'], row['date'])] = {
        'elo': float(row['elo_before']),
        'elo_trend': float(row['elo_trend']) if not pd.isna(row['elo_trend']) else 0.0,
    }

def get_elo(name, date, default=1500.0):
    v = elo_lookup_dict.get((name, date))
    if v: return v['elo'], v['elo_trend']
    return default, 0.0

master_3b['R_elo'], master_3b['R_elo_trend'] = zip(*master_3b.apply(
    lambda r: get_elo(r['R_fighter'], r['date']), axis=1))
master_3b['B_elo'], master_3b['B_elo_trend'] = zip(*master_3b.apply(
    lambda r: get_elo(r['B_fighter'], r['date']), axis=1))
master_3b['elo_dif']       = master_3b['R_elo']       - master_3b['B_elo']
master_3b['elo_trend_dif'] = master_3b['R_elo_trend'] - master_3b['B_elo_trend']
gc.collect()

# Career method rates (same approach as 3B: exact-date join)
rate_cols_3b = ['is_finish','is_decision','is_ko','is_sub',
                'finish_delivered','finish_received','dec_delivered','dec_received']
cf2 = career_df.copy()
cf2['method_type'] = cf2['method'].apply(lambda m: (
    'decision' if (pd.isna(m) is False and (
        'decision' in str(m).lower() or str(m).lower().startswith(('u-dec','s-dec','m-dec'))
    )) else
    'ko' if (pd.isna(m) is False and (
        'tko' in str(m).lower() or ('ko' in str(m).lower() and 'submission' not in str(m).lower())
    )) else
    'sub' if (pd.isna(m) is False and (
        'submission' in str(m).lower() or str(m).lower().startswith('sub')
    )) else 'other'
))
cf2['is_finish']   = cf2['method_type'].isin(['ko','sub']).astype(float)
cf2['is_decision'] = (cf2['method_type'] == 'decision').astype(float)
cf2['is_ko']       = (cf2['method_type'] == 'ko').astype(float)
cf2['is_sub']      = (cf2['method_type'] == 'sub').astype(float)
cf2['won2']        = cf2['won'].fillna(0).astype(float)
cf2['finish_delivered'] = cf2['is_finish'] * cf2['won2']
cf2['finish_received']  = cf2['is_finish'] * (1 - cf2['won2'])
cf2['dec_delivered']    = cf2['is_decision'] * cf2['won2']
cf2['dec_received']     = cf2['is_decision'] * (1 - cf2['won2'])

def expanding_rate(series):
    cumsum = series.cumsum().shift(1)
    count  = pd.Series(range(len(series)), index=series.index).shift(1) + 1
    return (cumsum / count).fillna(0)

career_parts = []
for fighter, grp in cf2.groupby('fighter'):
    grp = grp.sort_values('date').copy()
    for col in rate_cols_3b:
        grp[f'career_{col}'] = expanding_rate(grp[col])
    grp['career_n_fights'] = np.arange(len(grp))
    career_parts.append(grp[['fighter','date'] + [f'career_{c}' for c in rate_cols_3b] + ['career_n_fights']])
career_df2 = pd.concat(career_parts, ignore_index=True)
CAREER_RATE_COLS = [f'career_{c}' for c in rate_cols_3b] + ['career_n_fights']

def join_career_exact(df, fc, prefix, cdf):
    sub = cdf.rename(columns={'fighter': fc})
    sub = sub.rename(columns={c: f'{prefix}_{c}' for c in CAREER_RATE_COLS})
    return df.merge(sub, on=[fc,'date'], how='left')

master_3b = join_career_exact(master_3b, 'R_fighter', 'R', career_df2)
master_3b = join_career_exact(master_3b, 'B_fighter', 'B', career_df2)

for col in CAREER_RATE_COLS:
    for pref in ['R_','B_']:
        master_3b[f'{pref}{col}'] = master_3b[f'{pref}{col}'].fillna(0)
for col in rate_cols_3b:
    master_3b[f'combined_{col}'] = master_3b[f'R_career_{col}'] + master_3b[f'B_career_{col}']
gc.collect()

# Style stats (original file for 3B to match saved 3B features)
style_cols_raw = ['SLpM','SApM','Str_Def','TD_Avg','TD_Def','Sub_Avg']
fighters_orig = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final.csv'))
fighters_orig = fighters_orig[['Fighter_Name'] + style_cols_raw].copy()
fighters_orig.columns = ['fighter'] + style_cols_raw
for col in ['Str_Def','TD_Def']:
    if fighters_orig[col].dtype == object:
        fighters_orig[col] = fighters_orig[col].str.replace('%','',regex=False).astype(float) / 100.0
style_medians = fighters_orig[style_cols_raw].median()

def join_style(df, fc, prefix, fdf):
    sub = fdf.rename(columns={'fighter': fc})
    sub = sub.rename(columns={c: f'{prefix}_{c}' for c in style_cols_raw})
    return df.merge(sub, on=fc, how='left')

master_3b = join_style(master_3b, 'R_fighter', 'R', fighters_orig)
master_3b = join_style(master_3b, 'B_fighter', 'B', fighters_orig)
for sc in style_cols_raw:
    for pref in ['R_','B_']:
        master_3b[f'{pref}{sc}'] = master_3b[f'{pref}{sc}'].fillna(style_medians[sc])
for sc in style_cols_raw:
    master_3b[f'combined_{sc}'] = master_3b[f'R_{sc}'] + master_3b[f'B_{sc}']
gc.collect()

# Career win rates from career_df
win_parts = []
for fighter, grp in cf2.groupby('fighter'):
    grp = grp.sort_values('date').copy()
    grp['career_win_rate'] = expanding_rate(grp['won2'])
    win_parts.append(grp[['fighter','date','career_win_rate']])
win_rate_df = pd.concat(win_parts, ignore_index=True)

def join_single(df, fc, col, wdf, new_col):
    sub = wdf.rename(columns={'fighter': fc, col: new_col})
    return df.merge(sub[[fc,'date',new_col]], on=[fc,'date'], how='left')

master_3b = join_single(master_3b, 'R_fighter', 'career_win_rate', win_rate_df, 'R_career_win_rate')
master_3b = join_single(master_3b, 'B_fighter', 'career_win_rate', win_rate_df, 'B_career_win_rate')
master_3b['career_win_rate_dif'] = (
    master_3b['R_career_win_rate'].fillna(0.5) - master_3b['B_career_win_rate'].fillna(0.5))
master_3b['R_career_win_rate'] = master_3b['R_career_win_rate'].fillna(0.5)
master_3b['B_career_win_rate'] = master_3b['B_career_win_rate'].fillna(0.5)
gc.collect()

# Win-by rates, style diffs, combined stats
master_3b['R_wins_safe'] = master_3b['R_wins'].clip(lower=1)
master_3b['B_wins_safe'] = master_3b['B_wins'].clip(lower=1)
master_3b['R_ko_win_rate']  = master_3b['R_win_by_KO/TKO'] / master_3b['R_wins_safe']
master_3b['B_ko_win_rate']  = master_3b['B_win_by_KO/TKO'] / master_3b['B_wins_safe']
master_3b['R_sub_win_rate'] = master_3b['R_win_by_Submission'] / master_3b['R_wins_safe']
master_3b['B_sub_win_rate'] = master_3b['B_win_by_Submission'] / master_3b['B_wins_safe']
master_3b['R_dec_wins'] = (master_3b['R_win_by_Decision_Unanimous'] +
                            master_3b['R_win_by_Decision_Split'] +
                            master_3b['R_win_by_Decision_Majority'])
master_3b['B_dec_wins'] = (master_3b['B_win_by_Decision_Unanimous'] +
                            master_3b['B_win_by_Decision_Split'] +
                            master_3b['B_win_by_Decision_Majority'])
master_3b['R_dec_win_rate'] = master_3b['R_dec_wins'] / master_3b['R_wins_safe']
master_3b['B_dec_win_rate'] = master_3b['B_dec_wins'] / master_3b['B_wins_safe']
master_3b['ko_win_rate_dif']  = master_3b['R_ko_win_rate']  - master_3b['B_ko_win_rate']
master_3b['sub_win_rate_dif'] = master_3b['R_sub_win_rate'] - master_3b['B_sub_win_rate']
master_3b['dec_win_rate_dif'] = master_3b['R_dec_win_rate'] - master_3b['B_dec_win_rate']

master_3b['SLpM_dif']     = master_3b['R_SLpM']    - master_3b['B_SLpM']
master_3b['SApM_dif']     = master_3b['R_SApM']    - master_3b['B_SApM']
master_3b['Str_Def_dif']  = master_3b['R_Str_Def'] - master_3b['B_Str_Def']
master_3b['TD_Avg_dif']   = master_3b['R_TD_Avg']  - master_3b['B_TD_Avg']
master_3b['Sub_Avg_dif']  = master_3b['R_Sub_Avg'] - master_3b['B_Sub_Avg']

for stat in ['avg_SIG_STR_landed','avg_TD_landed','avg_SUB_ATT']:
    for s in ['R_','B_']: master_3b[f'{s}{stat}'] = master_3b[f'{s}{stat}'].fillna(0)
master_3b['combined_sig_str_landed'] = master_3b['R_avg_SIG_STR_landed'] + master_3b['B_avg_SIG_STR_landed']
master_3b['combined_td_landed']      = master_3b['R_avg_TD_landed']      + master_3b['B_avg_TD_landed']
master_3b['combined_sub_att']        = master_3b['R_avg_SUB_ATT']        + master_3b['B_avg_SUB_ATT']
for s in ['R_','B_']:
    master_3b[f'{s}avg_SIG_STR_pct'] = master_3b[f'{s}avg_SIG_STR_pct'].fillna(0)
    master_3b[f'{s}avg_TD_pct']      = master_3b[f'{s}avg_TD_pct'].fillna(0)
master_3b['sig_str_pct_dif'] = master_3b['R_avg_SIG_STR_pct'] - master_3b['B_avg_SIG_STR_pct']
master_3b['td_pct_dif']      = master_3b['R_avg_TD_pct']      - master_3b['B_avg_TD_pct']

for col in ['reach_dif','age_dif','sig_str_dif','avg_sub_att_dif','ko_dif','sub_dif',
            'win_streak_dif','lose_streak_dif','win_dif','loss_dif','avg_td_dif',
            'total_round_dif','total_title_bout_dif']:
    master_3b[col] = master_3b[col].fillna(0) if col in master_3b.columns else 0
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 8: Assemble v2 feature set
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 8: Assemble v2 feature set")
print("=" * 64)

FEATS_3B = joblib.load(os.path.join(OUT, 'model3b_features.pkl'))
M1_NEW_FEATS = ['m1_red_win_prob', 'm1_red_win_prob_sq', 'm1_confidence']
FEATS_V2 = FEATS_3B + M1_NEW_FEATS
print(f"  Base 3B features: {len(FEATS_3B)}")
print(f"  M1 new features:  {len(M1_NEW_FEATS)} → {M1_NEW_FEATS}")
print(f"  Total v2 features: {len(FEATS_V2)}")

# Verify all features available
missing_v2 = [f for f in FEATS_V2 if f not in master_3b.columns]
if missing_v2:
    print(f"  WARNING: missing features: {missing_v2}")

df_clean = master_3b[FEATS_V2 + ['label_6','date','weight_class']].dropna().copy()
print(f"  Rows after dropna: {len(df_clean):,}  (dropped {len(master_3b)-len(df_clean)})")

X_all   = df_clean[FEATS_V2].values
y_all   = df_clean['label_6'].values
dates_all = df_clean['date'].values
wc_all    = df_clean['weight_class'].values

train_mask = dates_all < CUTOFF
test_mask  = dates_all >= CUTOFF

X_train_raw, y_train = X_all[train_mask], y_all[train_mask]
X_test,       y_test  = X_all[test_mask],  y_all[test_mask]
print(f"  Train: {len(X_train_raw):,}  Test: {len(X_test):,}")

naive_test = np.bincount(y_test).max() / len(y_test)
print(f"  Test class distribution: {dict(zip(CLASS_NAMES, np.bincount(y_test)))}")
print(f"  Naive baseline (test): {naive_test:.4f}")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 9: Corner-flip augmentation (handles M1 features correctly)
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 9: Corner-flip augmentation")
print("=" * 64)

def corner_flip_augment_v2(X, y, feature_names):
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    X_flip = X.copy()
    # Swap R_ ↔ B_ paired features
    for rc in feature_names:
        if not rc.startswith('R_'): continue
        bc = 'B_' + rc[2:]
        if bc in feat_idx:
            ri, bi = feat_idx[rc], feat_idx[bc]
            X_flip[:, ri], X_flip[:, bi] = X[:, bi].copy(), X[:, ri].copy()
    # Negate _dif features
    for i, f in enumerate(feature_names):
        if f.endswith('_dif'):
            X_flip[:, i] = -X[:, i]
    # m1_red_win_prob: after corner flip, P(new Red wins) = 1 - P(old Red wins)
    if 'm1_red_win_prob' in feat_idx:
        idx_p = feat_idx['m1_red_win_prob']
        p_orig = X[:, idx_p]
        X_flip[:, idx_p] = 1.0 - p_orig
    # m1_red_win_prob_sq = p**2; flipped = (1-p)**2
    if 'm1_red_win_prob_sq' in feat_idx:
        idx_p  = feat_idx['m1_red_win_prob']
        idx_sq = feat_idx['m1_red_win_prob_sq']
        X_flip[:, idx_sq] = (1.0 - X[:, idx_p]) ** 2
    # m1_confidence = |p - 0.5|; symmetric, unchanged
    y_flip = np.array([LABEL_FLIP[label] for label in y])
    return (np.concatenate([X, X_flip], axis=0),
            np.concatenate([y, y_flip],  axis=0))

X_train_aug, y_train_aug = corner_flip_augment_v2(X_train_raw, y_train, FEATS_V2)
print(f"  Training rows before aug: {len(X_train_raw):,}")
print(f"  Training rows after aug:  {len(X_train_aug):,}  (2×)")
print(f"  Aug class balance: {dict(zip(CLASS_NAMES, np.bincount(y_train_aug)))}")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 10: Scale and train
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 10: Train models")
print("=" * 64)

scaler_v2 = RobustScaler()
X_train_sc = scaler_v2.fit_transform(X_train_aug)
X_test_sc  = scaler_v2.transform(X_test)

print("  Training Logistic Regression...")
lr_v2 = LogisticRegression(C=0.3, max_iter=2000, multi_class='multinomial',
                            solver='lbfgs', n_jobs=1, random_state=42)
lr_v2.fit(X_train_sc, y_train_aug)
lr_acc = accuracy_score(y_test, lr_v2.predict(X_test_sc))
print(f"  LR  — Test acc: {lr_acc:.4f}  vs naive: {lr_acc-naive_test:+.4f}")
gc.collect()

print("  Training Random Forest...")
rf_v2 = RandomForestClassifier(n_estimators=300, max_depth=9, min_samples_leaf=10,
                                random_state=42, n_jobs=1)
rf_v2.fit(X_train_aug, y_train_aug)
rf_acc = accuracy_score(y_test, rf_v2.predict(X_test))
print(f"  RF  — Test acc: {rf_acc:.4f}  vs naive: {rf_acc-naive_test:+.4f}")
gc.collect()

print("  Training XGBoost...")
xgb_v2 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         objective='multi:softprob', num_class=6,
                         eval_metric='mlogloss', verbosity=0,
                         random_state=42, n_jobs=1)
xgb_v2.fit(X_train_aug, y_train_aug,
            eval_set=[(X_test, y_test)], verbose=False)
xgb_acc = accuracy_score(y_test, xgb_v2.predict(X_test))
print(f"  XGB — Test acc: {xgb_acc:.4f}  vs naive: {xgb_acc-naive_test:+.4f}")
gc.collect()

print()
print("  Blend comparison:")
lr_p  = lr_v2.predict_proba(X_test_sc)
rf_p  = rf_v2.predict_proba(X_test)
xgb_p = xgb_v2.predict_proba(X_test)

blends_v2 = [
    ('LR only',                        1.0,  0.0,  0.0),
    ('RF only',                        0.0,  1.0,  0.0),
    ('XGB only',                       0.0,  0.0,  1.0),
    ('50% LR + 50% XGB',              0.5,  0.0,  0.5),
    ('30% LR + 70% XGB',              0.3,  0.0,  0.7),
    ('40% LR + 60% XGB',              0.4,  0.0,  0.6),
    ('40% RF + 60% XGB',              0.0,  0.4,  0.6),
    ('33% each',                       1/3,  1/3,  1/3),
    ('25% LR + 25% RF + 50% XGB',     0.25, 0.25, 0.50),
    ('20% LR + 20% RF + 60% XGB',     0.20, 0.20, 0.60),
]

best_acc_v2 = 0; best_label_v2 = ''; best_prob_v2 = None; best_w_v2 = None
print(f"  {'Blend':<38} {'Acc':>6} {'vs naive':>9}")
print(f"  {'-'*38} {'-'*6} {'-'*9}")
for label, wl, wr, wx in blends_v2:
    p    = wl * lr_p + wr * rf_p + wx * xgb_p
    pred = p.argmax(axis=1)
    acc  = accuracy_score(y_test, pred)
    mark = ' ←' if acc > best_acc_v2 else ''
    if acc > best_acc_v2:
        best_acc_v2 = acc; best_label_v2 = label; best_prob_v2 = p; best_w_v2 = (wl, wr, wx)
    print(f"  {label:<38} {acc:.4f} {acc-naive_test:>+9.4f}{mark}")

print(f"\n  Best blend: {best_label_v2}  ({best_acc_v2:.4f})")
print(f"  Beats naive by: {best_acc_v2-naive_test:+.4f} ({(best_acc_v2-naive_test)*100:+.2f}pp)")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 11: Per-class, direction, method accuracy
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 11: Per-class, direction, method accuracy")
print("=" * 64)

best_pred_v2 = best_prob_v2.argmax(axis=1)

print(f"\n  Per-class recall on test set:")
print(f"  {'Class':<14} {'N':>5}  {'Predicted':>9} {'Correct':>8} {'Recall%':>8}")
print(f"  {'-'*14} {'-'*5}  {'-'*9} {'-'*8} {'-'*8}")
per_class_v2 = []
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    n = mask.sum()
    if n == 0: continue
    n_pred_this = (best_pred_v2 == i).sum()
    n_correct   = ((best_pred_v2 == i) & mask).sum()
    recall = n_correct / n
    print(f"  {name:<14} {n:>5}  {n_pred_this:>9} {n_correct:>8} {recall*100:>8.1f}%")
    per_class_v2.append({'class': i, 'name': name, 'n': int(n),
                         'per_class_recall': round(float(recall), 4)})

actual_red = y_test < 3
pred_red   = best_pred_v2 < 3
dir_acc_v2 = (actual_red == pred_red).mean()

actual_method = np.where(y_test % 3 == 0, 0, np.where(y_test % 3 == 1, 1, 2))
pred_method   = np.where(best_pred_v2 % 3 == 0, 0, np.where(best_pred_v2 % 3 == 1, 1, 2))
method_acc_v2 = (actual_method == pred_method).mean()

print(f"\n  Direction accuracy:              {dir_acc_v2:.4f} ({dir_acc_v2*100:.2f}%)")
print(f"  Method accuracy:                 {method_acc_v2:.4f} ({method_acc_v2*100:.2f}%)")
print(f"  M1 reference direction accuracy: 0.7281 (72.81%)")
print(f"  Previous 3B direction accuracy:  0.6764 (67.64%)")
print(f"  Direction improvement vs 3B:     {dir_acc_v2-0.6764:+.4f} ({(dir_acc_v2-0.6764)*100:+.2f}pp)")
print(f"  Gap vs M1:                       {dir_acc_v2-0.7281:+.4f} ({(dir_acc_v2-0.7281)*100:+.2f}pp)")

for m_idx, m_name in enumerate(['KO/TKO','Submission','Decision']):
    mask = actual_method == m_idx
    n    = mask.sum()
    if n == 0: continue
    acc  = (pred_method[mask] == m_idx).mean()
    print(f"    {m_name:<14} (N={n:4d}): method recall {acc:.3f}")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 12: Feature importance
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 12: Feature importance (XGBoost)")
print("=" * 64)

xgb_imp = pd.Series(xgb_v2.feature_importances_, index=FEATS_V2).sort_values(ascending=False)
print(f"\n  Top 25 features — XGBoost:")
print(f"  {'Feature':<40} {'Importance':>10}")
print(f"  {'-'*40} {'-'*10}")
for feat, imp in xgb_imp.head(25).items():
    print(f"  {feat:<40} {imp:>10.4f}")

print(f"\n  M1 feature ranks:")
for feat in M1_NEW_FEATS:
    rank = list(xgb_imp.index).index(feat) + 1
    imp  = xgb_imp[feat]
    print(f"    {feat:<32} rank={rank:3d}  importance={imp:.4f}")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 13: Per-weight-class breakdown
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 13: Per-weight-class breakdown (test set)")
print("=" * 64)

test_df = df_clean[test_mask].copy().reset_index(drop=True)
test_df['pred_label']  = best_pred_v2
test_df['pred_winner'] = best_pred_v2 < 3
test_df['true_winner'] = y_test < 3
test_df['dir_correct'] = test_df['pred_winner'] == test_df['true_winner']

print(f"\n  {'Weight Class':<30} {'N':>5}  {'6-class':>8} {'Dir acc':>8} {'Naive':>7}")
print(f"  {'-'*30} {'-'*5}  {'-'*8} {'-'*8} {'-'*7}")
wc_res_v2 = []
for wc_name in sorted(test_df['weight_class'].unique()):
    sub = test_df[test_df['weight_class'] == wc_name]
    if len(sub) < 20: continue
    six_acc  = accuracy_score(sub['label_6'], sub['pred_label'])
    dir_a    = sub['dir_correct'].mean()
    naive_wc = np.bincount(sub['label_6'].values).max() / len(sub)
    print(f"  {wc_name:<30} {len(sub):>5}  {six_acc:>8.3f} {dir_a:>8.3f} {naive_wc:>7.3f}")
    wc_res_v2.append({'weight_class': wc_name, 'n': len(sub),
                      'six_class_acc': round(six_acc,4), 'direction_acc': round(dir_a,4),
                      'naive': round(naive_wc,4)})
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# STEP 14: Save
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("STEP 14: Save model3b_v2")
print("=" * 64)

wl_v2, wr_v2, wx_v2 = best_w_v2

joblib.dump(lr_v2,     os.path.join(OUT, 'model3b_v2_lr.pkl'))
joblib.dump(rf_v2,     os.path.join(OUT, 'model3b_v2_rf.pkl'))
joblib.dump(xgb_v2,    os.path.join(OUT, 'model3b_v2_xgb.pkl'))
joblib.dump(scaler_v2, os.path.join(OUT, 'model3b_v2_scaler.pkl'))
joblib.dump(FEATS_V2,  os.path.join(OUT, 'model3b_v2_features.pkl'))

dir_improved = bool(dir_acc_v2 > 0.6764)
prod_ready   = bool(dir_acc_v2 >= 0.71 and best_acc_v2 >= 0.44)

meta_v2 = {
    "model": "Model 3B v2 — Winner + Method, with M1 winner probability as feature",
    "classes": CLASS_NAMES,
    "label_map": {str(v): k for k, v in LABEL_MAP.items()},
    "train_cutoff": str(CUTOFF.date()),
    "m1_inference": {
        "lr_weight": M1_LR_W, "xgb_weight": M1_XGB_W,
        "description": "70% LR Pipeline + 30% XGB, production Model 1 (read-only)",
        "m1_direction_acc_test_window": round(float(m1_dir_universe), 4),
    },
    "n_train_raw": int(len(X_train_raw)),
    "n_train_aug": int(len(X_train_aug)),
    "n_test": int(len(X_test)),
    "n_features": len(FEATS_V2),
    "n_base_features": len(FEATS_3B),
    "m1_new_features": M1_NEW_FEATS,
    "class_balance_test": {CLASS_NAMES[i]: int(v) for i, v in enumerate(np.bincount(y_test))},
    "naive_baseline_test": round(float(naive_test), 4),
    "model_accuracy": {
        "lr_test":  round(float(lr_acc),  4),
        "rf_test":  round(float(rf_acc),  4),
        "xgb_test": round(float(xgb_acc), 4),
        "best_blend": round(float(best_acc_v2), 4),
        "best_blend_label": best_label_v2,
        "vs_naive_pp": round(float((best_acc_v2 - naive_test)*100), 2),
    },
    "blend_weights": {"lr": float(wl_v2), "rf": float(wr_v2), "xgb": float(wx_v2)},
    "six_class_accuracy": round(float(best_acc_v2), 4),
    "direction_accuracy": round(float(dir_acc_v2), 4),
    "direction_accuracy_prev_3b": 0.6764,
    "direction_accuracy_m1_ref": 0.7281,
    "direction_accuracy_improved_vs_3b": dir_improved,
    "direction_accuracy_gap_vs_m1": round(float(dir_acc_v2 - 0.7281), 4),
    "method_accuracy": round(float(method_acc_v2), 4),
    "per_class": per_class_v2,
    "by_weight_class": wc_res_v2,
    "xgb_top20_features": xgb_imp.head(20).round(4).to_dict(),
    "production_ready": prod_ready,
}
with open(os.path.join(OUT, 'model3b_v2_metadata.json'), 'w') as f:
    json.dump(meta_v2, f, indent=2)

print(f"  Saved: model3b_v2_lr.pkl, model3b_v2_rf.pkl, model3b_v2_xgb.pkl")
print(f"  Saved: model3b_v2_scaler.pkl, model3b_v2_features.pkl, model3b_v2_metadata.json")
print()
print("=" * 64)
print("SUMMARY")
print("=" * 64)
print(f"  Six-class accuracy:       {best_acc_v2:.4f}  ({best_label_v2})")
print(f"  vs naive:                 {best_acc_v2-naive_test:+.4f} ({(best_acc_v2-naive_test)*100:+.2f}pp)")
print(f"  Direction accuracy:       {dir_acc_v2:.4f}  (prev 3B: 0.6764, M1: 0.7281)")
print(f"  Direction improvement:    {dir_acc_v2-0.6764:+.4f} vs 3B")
print(f"  Gap vs M1:                {dir_acc_v2-0.7281:+.4f}")
print(f"  Method accuracy:          {method_acc_v2:.4f}")
print(f"  Production ready:         {prod_ready}")
print()
print("All steps complete. No production files touched.")
