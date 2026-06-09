#!/usr/bin/env python3
"""
build_value_bet_log_womens.py
Build historical value_bet_log for women's fights (2018+), equivalent to
data/value_bet_log.csv but using women's M1 as the predictor.

Steps:
  1  Data overview — women's fights in ufc-master.csv
  2  Run women's M1 retroactively (same preprocessing as training script)
  3  Pull Vegas odds → no-vig implied probs
  4  Compute pick, gap, zones, agreement types, pick_won
  5  Save data/value_bet_log_womens.csv + print summary
"""
import gc, os, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

os.chdir('/Users/allenthompson/Desktop/ufc-predictor')

# ─── Config ───────────────────────────────────────────────────────────────────
LOG_START     = '2018-01-01'
TRAIN_CUTOFF  = pd.Timestamp('2024-01-01')
LR_W, XGB_W   = 0.70, 0.30
WOMENS_CLASSES = [
    "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
]
WC_ORDER_WOMENS = {
    "Women's Strawweight": 0, "Women's Flyweight": 1,
    "Women's Bantamweight": 2, "Women's Featherweight": 3,
}

print("=" * 70)
print("BUILDING WOMEN'S VALUE BET LOG")
print("=" * 70)

# ─── STEP 1 — Data overview ───────────────────────────────────────────────────
print("\n── STEP 1: Data overview ──────────────────────────────────────────────")
df_all = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_all['date'] = pd.to_datetime(df_all['date'])

womens_all = df_all[df_all['weight_class'].isin(WOMENS_CLASSES)].copy()
print(f"Total women's rows in ufc-master.csv : {len(womens_all):,}")
print(f"\nBy weight class:")
for wc, n in womens_all['weight_class'].value_counts().items():
    print(f"  {wc:<30}: {n:,}")
print(f"\nDate range: {womens_all['date'].min().date()} → {womens_all['date'].max().date()}")
gc.collect()

# ─── STEP 2 — Build women's M1 feature matrix ────────────────────────────────
print("\n── STEP 2: Build features + run Women's M1 ───────────────────────────")

# Load saved model + feature columns
print("Loading women's M1 models...")
m1_lr    = joblib.load('model/ufc_model_womens_lr.pkl')
m1_xgb   = joblib.load('model/ufc_model_womens_xgb.pkl')
m1_feats = joblib.load('model/ufc_model_womens_features.pkl')
print(f"  {len(m1_feats)} features  |  blend: {LR_W*100:.0f}% LR + {XGB_W*100:.0f}% XGB")

# Filter: women's, 2018+, valid winner, ML odds non-null
df = df_all[
    df_all['weight_class'].isin(WOMENS_CLASSES) &
    (df_all['date'] >= LOG_START) &
    df_all['Winner'].isin(['Red', 'Blue']) &
    df_all['R_odds'].notna() & df_all['B_odds'].notna() &
    (df_all['R_odds'] != 0) & (df_all['B_odds'] != 0)
].copy().reset_index(drop=True)
df = df.sort_values('date').reset_index(drop=True)
print(f"\nFights in scope (2018+, valid winner, ML odds): {len(df):,}")

# ── Career stats ──────────────────────────────────────────────────────────────
print("Computing career stats (shift=1, no leakage)...")
career_raw = pd.read_csv('data/career_fights_updated.csv')
career_raw['date'] = pd.to_datetime(career_raw['date'])
career_raw = career_raw.sort_values(['fighter','date']).reset_index(drop=True)

all_womens = df_all[df_all['weight_class'].isin(WOMENS_CLASSES) &
                    df_all['Winner'].isin(['Red','Blue'])].copy()
womens_fighters = set(all_womens['R_fighter']) | set(all_womens['B_fighter'])
career_w = career_raw[career_raw['fighter'].isin(womens_fighters)].copy()
career_w = career_w.sort_values(['fighter','date']).reset_index(drop=True)

def compute_career_stats(cdf):
    df_c = cdf.copy()
    df_c['_ko']  = ((df_c['won']==1) & df_c['method'].str.contains('KO|TKO', case=False, na=False)).astype(float)
    df_c['_sub'] = ((df_c['won']==1) & df_c['method'].str.contains('Sub|Submission', case=False, na=False)).astype(float)
    df_c['_fin'] = ((df_c['won']==1) & df_c['method'].str.contains('KO|TKO|Sub', case=False, na=False)).astype(float)

    g = df_c.groupby('fighter', sort=False)
    df_c['cum_fights'] = g.cumcount()

    for src, dst in [('won','_cs_won'),('_ko','_cs_ko'),('_sub','_cs_sub'),('_fin','_cs_fin')]:
        df_c[dst] = g[src].cumsum() - df_c[src]

    safe_n = df_c['cum_fights'].clip(lower=1)
    df_c['career_win_rate']    = np.where(df_c['cum_fights']>0, df_c['_cs_won']/safe_n, 0.5)
    df_c['ko_finish_rate']     = np.where(df_c['cum_fights']>0, df_c['_cs_ko']/safe_n,  0.0)
    df_c['sub_finish_rate']    = np.where(df_c['cum_fights']>0, df_c['_cs_sub']/safe_n, 0.0)
    df_c['career_finish_rate'] = np.where(df_c['cum_fights']>0, df_c['_cs_fin']/safe_n, 0.0)

    def _roll(series, w, default):
        return series.shift(1).rolling(w, min_periods=1).mean().fillna(default)

    df_c['last3_win_rate']    = g['won'].transform(lambda x: _roll(x, 3,  0.5))
    df_c['last10_win_rate']   = g['won'].transform(lambda x: _roll(x, 10, 0.5))
    df_c['last5_won']         = g['won'].transform(lambda x: _roll(x, 5,  0.5))
    df_c['last5_finish_rate'] = g['_fin'].transform(lambda x: _roll(x, 5, 0.0))
    df_c['trend_score']       = df_c['last3_win_rate'] - df_c['last10_win_rate']
    df_c['_prev_date']        = g['date'].transform(lambda x: x.shift(1))
    df_c['layoff_days']       = (df_c['date'] - df_c['_prev_date']).dt.days.fillna(180.0).clip(lower=0)

    all_wr = {f: grp['won'].sum()/max(1,len(grp)) for f, grp in career_w.groupby('fighter')}
    opp_col = df_c['opponent'].tolist()
    ftr_col = df_c['fighter'].tolist()
    ftr_pos = defaultdict(list)
    for pos, idx in enumerate(df_c.index.tolist()):
        ftr_pos[ftr_col[pos]].append(pos)

    oq = np.full(len(df_c), 0.5)
    for ftr, positions in ftr_pos.items():
        for rank, pos in enumerate(positions):
            past = [opp_col[p] for p in positions[max(0,rank-5):rank]]
            rates = [all_wr.get(o, 0.5) for o in past]
            oq[pos] = float(np.mean(rates)) if rates else 0.5
    df_c['opp_quality'] = oq

    keep_cols = ['fighter','date','cum_fights','career_win_rate','ko_finish_rate',
                 'sub_finish_rate','career_finish_rate','last3_win_rate','last10_win_rate',
                 'last5_won','last5_finish_rate','trend_score','layoff_days','opp_quality']
    return df_c[keep_cols]

career_stats = compute_career_stats(career_w)
print(f"  Career stat rows: {len(career_stats):,}")
gc.collect()

# ── Elo (women's only, K=48) ──────────────────────────────────────────────────
print("Computing women's Elo (K=48, base=1500)...")

def compute_elo_womens(df_src):
    df_s = df_src.sort_values('date').reset_index(drop=True)
    elo = {}; hist = []
    for _, row in df_s.iterrows():
        r, b, date, winner = row['R_fighter'], row['B_fighter'], row['date'], row['Winner']
        r_b = elo.get(r, 1500.0); b_b = elo.get(b, 1500.0)
        r_exp = 1.0 / (1.0 + 10.0**((b_b-r_b)/400.0))
        r_act = 1.0 if winner=='Red' else (0.0 if winner=='Blue' else 0.5)
        b_act = 1.0 - r_act
        r_a = r_b + 48*(r_act-r_exp); b_a = b_b + 48*(b_act-(1-r_exp))
        hist.append({'fighter':r,'date':date,'elo_before':r_b,'elo_after':r_a})
        hist.append({'fighter':b,'date':date,'elo_before':b_b,'elo_after':b_a})
        elo[r]=r_a; elo[b]=b_a
    elo_df = pd.DataFrame(hist).sort_values(['fighter','date']).reset_index(drop=True)
    elo_df['elo_trend'] = elo_df.groupby('fighter')['elo_before'].transform(lambda x: x - x.shift(3))
    return elo_df

elo_hist = compute_elo_womens(all_womens)
print(f"  Elo history rows: {len(elo_hist):,}")
gc.collect()

# ── Style stats ───────────────────────────────────────────────────────────────
style_df = pd.read_csv('data/ufc_fighters_final_updated.csv')
for col in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    style_df[col] = pd.to_numeric(
        style_df[col].astype(str).str.replace('%','',regex=False),
        errors='coerce').fillna(0.0) / 100.0
style_df = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last')
style_cols = ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']

# ── Merge career stats, Elo, style ───────────────────────────────────────────
print("Merging career stats, Elo, style stats via merge_asof...")
career_cols = [c for c in career_stats.columns if c not in ('fighter','date')]
r_career = career_stats.rename(columns={'fighter':'R_fighter',
                                         **{c:f'R_{c}' for c in career_cols}})
b_career = career_stats.rename(columns={'fighter':'B_fighter',
                                         **{c:f'B_{c}' for c in career_cols}})
df = pd.merge_asof(df.sort_values('date'), r_career.sort_values('date'),
                   on='date', by='R_fighter', direction='backward')
df = pd.merge_asof(df.sort_values('date'), b_career.sort_values('date'),
                   on='date', by='B_fighter', direction='backward')

career_defaults = {'cum_fights':0,'career_win_rate':0.5,'ko_finish_rate':0.0,
                   'sub_finish_rate':0.0,'career_finish_rate':0.0,
                   'last3_win_rate':0.5,'last10_win_rate':0.5,
                   'last5_won':0.5,'last5_finish_rate':0.0,
                   'trend_score':0.0,'layoff_days':180.0,'opp_quality':0.5}
for stat, default in career_defaults.items():
    df[f'R_{stat}'] = df[f'R_{stat}'].fillna(default)
    df[f'B_{stat}'] = df[f'B_{stat}'].fillna(default)

elo_cols_df = elo_hist[['fighter','date','elo_before','elo_trend']].copy()
elo_r = elo_cols_df.rename(columns={'fighter':'R_fighter','elo_before':'R_elo','elo_trend':'R_elo_trend'})
elo_b = elo_cols_df.rename(columns={'fighter':'B_fighter','elo_before':'B_elo','elo_trend':'B_elo_trend'})
df = pd.merge_asof(df.sort_values('date'), elo_r.sort_values('date'),
                   on='date', by='R_fighter', direction='backward')
df = pd.merge_asof(df.sort_values('date'), elo_b.sort_values('date'),
                   on='date', by='B_fighter', direction='backward')
df['R_elo'] = df['R_elo'].fillna(1500.0)
df['B_elo'] = df['B_elo'].fillna(1500.0)
df['R_elo_trend'] = df['R_elo_trend'].fillna(0.0)
df['B_elo_trend'] = df['B_elo_trend'].fillna(0.0)

r_style = style_df[['Fighter_Name']+style_cols].rename(
    columns={'Fighter_Name':'R_fighter',**{c:f'R_{c}' for c in style_cols}})
b_style = style_df[['Fighter_Name']+style_cols].rename(
    columns={'Fighter_Name':'B_fighter',**{c:f'B_{c}' for c in style_cols}})
df = df.merge(r_style, on='R_fighter', how='left')
df = df.merge(b_style, on='B_fighter', how='left')
for col in [f'{p}{s}' for p in ('R_','B_') for s in style_cols]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
gc.collect()

# ── Engineer derived features ─────────────────────────────────────────────────
print("Engineering features...")
df['weight_class_ord'] = df['weight_class'].map(WC_ORDER_WOMENS).fillna(1).astype(int)
df['R_southpaw'] = (df['R_Stance'].str.lower()=='southpaw').astype(int)
df['B_southpaw'] = (df['B_Stance'].str.lower()=='southpaw').astype(int)
df['orth_clash']  = ((df['R_southpaw']==0)&(df['B_southpaw']==0)).astype(int)
df['south_clash'] = ((df['R_southpaw']==1)&(df['B_southpaw']==1)).astype(int)
df['R_age'] = pd.to_numeric(df['R_age'], errors='coerce').fillna(28.0)
df['B_age'] = pd.to_numeric(df['B_age'], errors='coerce').fillna(28.0)
df['R_age_x_exp']  = df['R_age'] * df['R_cum_fights']
df['B_age_x_exp']  = df['B_age'] * df['B_cum_fights']
df['age_x_exp_dif'] = df['R_age_x_exp'] - df['B_age_x_exp']

def layoff_buckets(prefix, days):
    d = days.fillna(180.0)
    return {f'{prefix}layoff_lt90': (d<90).astype(int),
            f'{prefix}layoff_90_180': ((d>=90)&(d<180)).astype(int),
            f'{prefix}layoff_180_365': ((d>=180)&(d<365)).astype(int),
            f'{prefix}layoff_gt365': (d>=365).astype(int)}
for k, v in {**layoff_buckets('R_', df['R_layoff_days']),
             **layoff_buckets('B_', df['B_layoff_days'])}.items():
    df[k] = v.values

for stat in ['career_win_rate','last5_won','last5_finish_rate','opp_quality',
             'trend_score','ko_finish_rate','sub_finish_rate',
             'last3_win_rate','last10_win_rate']:
    df[f'{stat}_dif'] = df[f'R_{stat}'] - df[f'B_{stat}']

for col in ['R_wins','R_losses','R_Height_cms','R_Reach_cms',
            'B_wins','B_losses','B_Height_cms','B_Reach_cms',
            'R_avg_SIG_STR_landed','R_avg_TD_landed','R_avg_SIG_STR_pct',
            'R_avg_SUB_ATT','R_avg_TD_pct',
            'B_avg_SIG_STR_landed','B_avg_TD_landed','B_avg_SIG_STR_pct',
            'B_avg_SUB_ATT','B_avg_TD_pct',
            'R_current_win_streak','R_current_lose_streak','R_longest_win_streak',
            'B_current_win_streak','B_current_lose_streak','B_longest_win_streak',
            'B_total_title_bouts']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

df['win_dif']        = df['R_wins'] - df['B_wins']
df['loss_dif']       = df['R_losses'] - df['B_losses']
df['win_streak_dif'] = df['R_current_win_streak'] - df['B_current_win_streak']
df['lose_streak_dif']= df['R_current_lose_streak']- df['B_current_lose_streak']
df['height_dif']     = df['R_Height_cms'] - df['B_Height_cms']
df['reach_dif']      = df['R_Reach_cms']  - df['B_Reach_cms']
df['age_dif']        = df['R_age'] - df['B_age']
df['sig_str_dif']    = df['R_avg_SIG_STR_landed'] - df['B_avg_SIG_STR_landed']
df['avg_td_dif']     = df['R_avg_TD_landed'] - df['B_avg_TD_landed']
df['ko_dif']         = df['R_ko_finish_rate'] - df['B_ko_finish_rate']
df['sub_dif']        = df['R_sub_finish_rate']- df['B_sub_finish_rate']
df['total_title_bout_dif'] = 0
df['SLpM_dif']   = df['R_SLpM']   - df['B_SLpM']
df['SApM_dif']   = df['R_SApM']   - df['B_SApM']
df['Str_Def_dif']= df['R_Str_Def']- df['B_Str_Def']
df['TD_Def_dif'] = df['R_TD_Def'] - df['B_TD_Def']
df['Sub_Avg_dif']= df['R_Sub_Avg']- df['B_Sub_Avg']
df['TD_Avg_dif'] = df['R_TD_Avg'] - df['B_TD_Avg']
df['elo_dif']       = df['R_elo']       - df['B_elo']
df['elo_trend_dif'] = df['R_elo_trend'] - df['B_elo_trend']

# Apply debut filter (same as M1 training)
n_before = len(df)
df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy().reset_index(drop=True)
print(f"  Debut filter removed: {n_before - len(df)}  |  Remaining: {len(df):,}")

# Build feature matrix — reindex to exact saved feature columns
X_df = df.reindex(columns=m1_feats).fillna(0.0)
X = X_df.values.astype(float)
X = np.nan_to_num(X, nan=0.0)
print(f"  Feature matrix: {X.shape}")

# Run women's M1
print("Running Women's M1 predictions...")
p_lr_all  = m1_lr.predict_proba(X)[:, 1]
p_xgb_all = m1_xgb.predict_proba(X)[:, 1]
m1_probs  = LR_W * p_lr_all + XGB_W * p_xgb_all
df['m1_prob'] = m1_probs

print(f"  M1 prob range: {m1_probs.min():.3f} – {m1_probs.max():.3f}  mean={m1_probs.mean():.3f}")
gc.collect()

# ─── STEP 3 — Vegas odds → no-vig ────────────────────────────────────────────
print("\n── STEP 3: Vegas odds → no-vig ───────────────────────────────────────")

def implied(ml):
    ml = pd.to_numeric(ml, errors='coerce')
    return np.where(ml.isna() | (ml==0), np.nan,
           np.where(ml < 0, (-ml)/(-ml+100), 100/(ml+100)))

r_raw = implied(df['R_odds'])
b_raw = implied(df['B_odds'])
total = r_raw + b_raw
total = np.where(total <= 0, 1.0, total)
r_novig = r_raw / total
b_novig = b_raw / total

df['r_novig'] = r_novig
df['b_novig'] = b_novig

# ─── STEP 4 — Pick, gap, zones, agreement ─────────────────────────────────────
print("── STEP 4: Pick, gap, zones, agreement type ──────────────────────────")

# M1 pick: red if m1_prob > 0.5
pick_red    = df['m1_prob'] > 0.5
m2a_pick    = np.where(pick_red, df['R_fighter'], df['B_fighter'])
pick_prob   = np.where(pick_red, df['m1_prob'], 1.0 - df['m1_prob'])
pick_novig  = np.where(pick_red, r_novig, b_novig)
closing_odds= np.where(pick_red,
                        pd.to_numeric(df['R_odds'], errors='coerce').values,
                        pd.to_numeric(df['B_odds'], errors='coerce').values)

gap     = pick_prob - pick_novig
gap_dir = np.where(gap > 0, 1, -1)

def gap_zone_num(g_abs):
    return np.select(
        [g_abs < 0.01, g_abs < 0.02, g_abs < 0.03, g_abs < 0.05,
         g_abs < 0.08, g_abs < 0.10],
        [0, 1, 2, 3, 4, 5],
        default=6
    )

def gap_zone_lbl(g_abs):
    return np.select(
        [g_abs < 0.01, g_abs < 0.02, g_abs < 0.03, g_abs < 0.05,
         g_abs < 0.08, g_abs < 0.10],
        ['Lock','Strong','Lean','Watch','Value','Strong Value'],
        default='Max Value'
    )

gap_abs  = np.abs(gap)
g_zone   = gap_zone_num(gap_abs)
g_label  = gap_zone_lbl(gap_abs)

# Vegas agree: M1 picks the Vegas favorite (negative ML = favorite)
r_is_fav   = pd.to_numeric(df['R_odds'], errors='coerce') < 0
vegas_agree = np.where(pick_red, r_is_fav.astype(int).values,
                        (~r_is_fav).astype(int).values)

# Agreement type (M1-only version; m1_m2a_agree = 1 always)
agreement_type = np.select(
    [
        (gap_dir == 1) & (vegas_agree == 0),   # CONFIRM_DOG
        (gap_dir == 1) & (vegas_agree == 1),   # CONFIRM_FAV
    ],
    ['CONFIRM_DOG', 'CONFIRM_FAV'],
    default='NO_EDGE'
)

# pick_won
target    = (df['Winner'] == 'Red').astype(int).values
pick_won  = np.where(pick_red, target, 1 - target)

# split label
split = np.where(df['date'] < TRAIN_CUTOFF, 'train', 'test')

# ─── STEP 5 — Assemble and save ───────────────────────────────────────────────
print("── STEP 5: Assemble log, validate, save ─────────────────────────────")

no_of_rounds = pd.to_numeric(df.get('no_of_rounds', pd.Series(3, index=df.index)),
                              errors='coerce').fillna(3).astype(int)

log = pd.DataFrame({
    'date':           df['date'].dt.strftime('%Y-%m-%d'),
    'f1_name':        df['R_fighter'].values,
    'f2_name':        df['B_fighter'].values,
    'weight_class':   df['weight_class'].values,
    'no_of_rounds':   no_of_rounds.values,
    'm1_prob':        np.round(df['m1_prob'].values * 100, 2),       # as percentage (like men's log)
    'm2a_prob':       np.round(df['m1_prob'].values * 100, 2),       # = m1 until women's M2A
    'm2a_pick':       m2a_pick,
    'pick_novig':     np.round(pick_novig * 100, 2),
    'gap':            np.round(gap * 100, 2),
    'gap_size':       np.round(gap_abs * 100, 2),
    'gap_zone':       g_zone,
    'gap_zone_label': g_label,
    'gap_direction':  gap_dir,
    'closing_odds':   closing_odds,
    'm1_m2a_agree':   1,                     # placeholder until women's M2A is integrated
    'vegas_agree':    vegas_agree,
    'triple_agree':   vegas_agree,            # = vegas_agree since m1_m2a_agree=1 always
    'm2b_win_prob':   np.nan,
    'm2b_confidence': np.nan,
    'pick_won':       pick_won,
    'split':          split,
    'agreement_type': agreement_type,
})

log['date'] = pd.to_datetime(log['date'])

# Drop rows with null no-vig (couldn't parse odds)
n_before = len(log)
log = log.dropna(subset=['pick_novig']).reset_index(drop=True)
print(f"  Rows with parseable odds: {len(log):,}  (dropped {n_before-len(log)} unparseable)")

log.to_csv('data/value_bet_log_womens.csv', index=False)
print(f"  Saved → data/value_bet_log_womens.csv  ({len(log):,} rows)")
gc.collect()

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WOMEN'S VALUE BET LOG — SUMMARY")
print("=" * 70)

print(f"\n  Total rows      : {len(log):,}")
print(f"  Date range      : {log['date'].min().date()} → {log['date'].max().date()}")
print(f"  Train (<2024)   : {(log['split']=='train').sum():,}")
print(f"  Test  (2024+)   : {(log['split']=='test').sum():,}")

print(f"\n  By weight class:")
for wc, n in log['weight_class'].value_counts().items():
    print(f"    {wc:<30}: {n:,}")

print(f"\n  Agreement type distribution:")
for at, n in log['agreement_type'].value_counts().items():
    pct = n/len(log)*100
    print(f"    {at:<15}: {n:4d}  ({pct:.1f}%)")

# Win rate + flat ROI per agreement type
def roi_flat(sub, stake=20):
    payouts = []
    for _, r in sub.iterrows():
        ml = r['closing_odds']
        if pd.isna(ml): continue
        if ml > 0:
            payouts.append(stake * ml/100 if r['pick_won']==1 else -stake)
        else:
            payouts.append(stake * 100/abs(ml) if r['pick_won']==1 else -stake)
    if not payouts: return None
    return sum(payouts) / (stake * len(payouts)) * 100

print(f"\n  Win rate and flat $20 ROI by agreement type:")
print(f"  {'Type':<15}  {'N':>5}  {'WR':>7}  {'ROI':>8}")
print(f"  {'-'*15}  {'-'*5}  {'-'*7}  {'-'*8}")
for at in ['CONFIRM_DOG','CONFIRM_FAV','NO_EDGE']:
    sub = log[log['agreement_type']==at]
    if len(sub) == 0: continue
    wr  = sub['pick_won'].mean()
    roi = roi_flat(sub)
    print(f"  {at:<15}  {len(sub):>5}  {wr*100:>6.1f}%  {roi:>+7.1f}%" if roi else
          f"  {at:<15}  {len(sub):>5}  {wr*100:>6.1f}%  {'N/A':>8}")

print(f"\n  Gap zone distribution:")
for lbl in ['Lock','Strong','Lean','Watch','Value','Strong Value','Max Value']:
    n = (log['gap_zone_label']==lbl).sum()
    pct = n/len(log)*100
    print(f"    {lbl:<14}: {n:4d}  ({pct:.1f}%)")

# Method odds availability
METHOD_COLS = ['r_ko_odds','b_ko_odds','r_sub_odds','b_sub_odds','r_dec_odds','b_dec_odds']
all_method_mask = df_all[
    df_all['weight_class'].isin(WOMENS_CLASSES) &
    (df_all['date'] >= LOG_START) &
    df_all['Winner'].isin(['Red','Blue']) &
    df_all['R_odds'].notna() & df_all['B_odds'].notna() &
    (df_all['R_odds'] != 0) & (df_all['B_odds'] != 0)
][METHOD_COLS].notna().all(axis=1)
n_method_odds = all_method_mask.sum()
print(f"\n  Method odds available (all 6 cols): {n_method_odds:,} / {len(log):,} "
      f"({n_method_odds/len(log)*100:.1f}%) — available for future women's M2A")

# Side-by-side vs men's
print(f"\n{'─'*70}")
print("  SIDE-BY-SIDE vs MEN'S LOG")
print(f"{'─'*70}")
try:
    men = pd.read_csv('data/value_bet_log.csv')
    men_wr  = men['pick_won'].mean()
    men_roi = roi_flat(men)
    w_wr    = log['pick_won'].mean()
    w_roi   = roi_flat(log)
    print(f"  {'Metric':<20}  {'Men':>9}  {'Women':>9}")
    print(f"  {'-'*20}  {'-'*9}  {'-'*9}")
    print(f"  {'N fights':<20}  {len(men):>9,}  {len(log):>9,}")
    print(f"  {'Overall win rate':<20}  {men_wr*100:>8.1f}%  {w_wr*100:>8.1f}%")
    print(f"  {'Flat $20 ROI':<20}  {men_roi:>+8.1f}%  {w_roi:>+8.1f}%")
    for at in ['CONFIRM_DOG','CONFIRM_FAV','NO_EDGE']:
        m_sub = men[men['m1_m2a_agree']==1]  # approximate for men's
        # For men's: derive agreement type from existing columns
        def men_agree_type(r):
            if r['m1_m2a_agree']==0: return 'SPLIT'
            if r['gap_direction']==1 and r['vegas_agree']==0: return 'CONFIRM_DOG'
            if r['gap_direction']==1 and r['vegas_agree']==1: return 'CONFIRM_FAV'
            return 'NO_EDGE'
        men['_at'] = men.apply(men_agree_type, axis=1)
        m_sub_at = men[men['_at']==at]
        w_sub_at = log[log['agreement_type']==at]
        m_wr_at = m_sub_at['pick_won'].mean() if len(m_sub_at)>0 else float('nan')
        w_wr_at = w_sub_at['pick_won'].mean() if len(w_sub_at)>0 else float('nan')
        m_roi_at = roi_flat(m_sub_at) or float('nan')
        w_roi_at = roi_flat(w_sub_at) or float('nan')
        print(f"  {at+' WR':<20}  {m_wr_at*100:>8.1f}%  {w_wr_at*100:>8.1f}%")
        print(f"  {at+' ROI':<20}  {m_roi_at:>+8.1f}%  {w_roi_at:>+8.1f}%")
    del men
except Exception as e:
    print(f"  (Could not load men's log for comparison: {e})")

print(f"\n{'='*70}")
print("DONE — data/value_bet_log_womens.csv saved")
print(f"{'='*70}")
