#!/usr/bin/env python3
"""
Step 4 — Retrain Model 1 with 3 Variants
A: current 114 features minus flagged removals (110 feats)
B: current 114 features plus new trajectory features (129 feats)
C: trimmed features plus new trajectory features (125 feats)
Saves: model/variant_A.pkl, model/variant_B.pkl, model/variant_C.pkl + feature lists
"""
import sys, os, warnings, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA  = os.path.join(ROOT, 'data')
MODEL = os.path.join(ROOT, 'model')
OUT   = os.path.join(ROOT, 'experiments', 'research')

TRAIN_START  = '2018-01-01'
TRAIN_CUTOFF = '2024-01-01'
LR_WEIGHT    = 0.90
XGB_WEIGHT   = 0.10

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

# Features flagged for removal in Step 1
REMOVAL_CANDIDATES = ['R_total_title_bouts', 'B_layoff_gt365', 'B_southpaw', 'title_bout_bin']
# Redundant pair — keep career_win_rate_dif (higher corr), drop last10_win_rate_dif
REDUNDANT_DROP     = ['last10_win_rate_dif']

# New trajectory features from Step 2 (drop avg_fights_between_losses — NaN issues)
NEW_TRAJ_FEATS = [
    'fights_since_finish',
    'win_rate_l5_vs_career',
    'finish_rate_trend',
    'longest_lose_streak_ever',
    'comeback_flag',
]

# ─── Helpers (copied from train_model1.py — no imports to avoid touching prod) ─

def compute_elo(df_all, K=48, base=1500.0):
    df_sorted = df_all.sort_values('date').reset_index(drop=True)
    elo = {}
    history_rows = []
    for _, row in df_sorted.iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        date, winner = row['date'], row['Winner']
        r_before = elo.get(r, base); b_before = elo.get(b, base)
        r_exp = 1.0 / (1.0 + 10.0 ** ((b_before - r_before) / 400.0))
        if winner == 'Red':   r_act, b_act = 1.0, 0.0
        elif winner == 'Blue': r_act, b_act = 0.0, 1.0
        else:                  r_act, b_act = 0.5, 0.5
        r_after = r_before + K * (r_act - r_exp)
        b_after = b_before + K * ((1 - r_act) - (1 - r_exp))
        history_rows.append({'fighter': r, 'opponent': b, 'date': date,
                              'elo_before': r_before, 'elo_after': r_after, 'result': r_act})
        history_rows.append({'fighter': b, 'opponent': r, 'date': date,
                              'elo_before': b_before, 'elo_after': b_after, 'result': 1-r_act})
        elo[r] = r_after; elo[b] = b_after
    hist = pd.DataFrame(history_rows).sort_values(['fighter','date']).reset_index(drop=True)
    hist['elo_trend'] = hist.groupby('fighter')['elo_before'].transform(lambda x: x - x.shift(3))
    return hist

def compute_career_stats(career_df, all_win_rates):
    df = career_df.sort_values(['fighter','date']).copy().reset_index(drop=True)
    df['_ko']  = ((df['won']==1) & df['method'].str.contains('KO|TKO',   case=False, na=False)).astype(float)
    df['_sub'] = ((df['won']==1) & df['method'].str.contains('Sub|Submission', case=False, na=False)).astype(float)
    df['_fin'] = ((df['won']==1) & df['method'].str.contains('KO|TKO|Sub', case=False, na=False)).astype(float)
    g = df.groupby('fighter', sort=False)
    df['cum_fights'] = g.cumcount()
    for src, dst in [('won','_cs_won'),('_ko','_cs_ko'),('_sub','_cs_sub'),('_fin','_cs_fin')]:
        df[dst] = g[src].cumsum() - df[src]
    safe_n = df['cum_fights'].clip(lower=1)
    df['career_win_rate']    = np.where(df['cum_fights']>0, df['_cs_won']/safe_n, 0.5)
    df['ko_finish_rate']     = np.where(df['cum_fights']>0, df['_cs_ko'] /safe_n, 0.0)
    df['sub_finish_rate']    = np.where(df['cum_fights']>0, df['_cs_sub']/safe_n, 0.0)
    df['career_finish_rate'] = np.where(df['cum_fights']>0, df['_cs_fin']/safe_n, 0.0)
    def _roll(series, w, default):
        return series.shift(1).rolling(w, min_periods=1).mean().fillna(default)
    df['last3_win_rate']    = g['won'].transform(lambda x: _roll(x, 3, 0.5))
    df['last10_win_rate']   = g['won'].transform(lambda x: _roll(x, 10, 0.5))
    df['last5_won']         = g['won'].transform(lambda x: _roll(x, 5, 0.5))
    df['last5_finish_rate'] = g['_fin'].transform(lambda x: _roll(x, 5, 0.0))
    df['trend_score']       = df['last3_win_rate'] - df['last10_win_rate']
    df['_prev_date'] = g['date'].transform(lambda x: x.shift(1))
    df['layoff_days'] = (df['date'] - df['_prev_date']).dt.days.fillna(180.0).clip(lower=0)
    opp_col = df['opponent'].tolist(); fighter_col = df['fighter'].tolist()
    fighter_positions = defaultdict(list)
    for pos in range(len(fighter_col)):
        fighter_positions[fighter_col[pos]].append(pos)
    opp_quality_arr = np.full(len(df), 0.5)
    for fighter, positions in fighter_positions.items():
        for rank, pos in enumerate(positions):
            past_opps = [opp_col[p] for p in positions[max(0,rank-5):rank]]
            rates = [all_win_rates[opp] for opp in past_opps if opp in all_win_rates]
            opp_quality_arr[pos] = float(np.mean(rates)) if rates else 0.5
    df['opp_quality'] = opp_quality_arr
    df.drop(columns=['_ko','_sub','_fin','_cs_won','_cs_ko','_cs_sub','_cs_fin','_prev_date'], inplace=True)
    return df[['fighter','date','cum_fights','career_win_rate','ko_finish_rate','sub_finish_rate',
               'career_finish_rate','last3_win_rate','last10_win_rate','last5_won',
               'last5_finish_rate','trend_score','layoff_days','opp_quality']]

def compute_trajectory_stats(career_df):
    """Trajectory features per-fighter per-fight."""
    df = career_df.sort_values(['fighter','date']).copy().reset_index(drop=True)
    df['_ko']  = ((df['won']==1) & df['method'].str.contains('KO|TKO',   case=False, na=False)).astype(float)
    df['_sub'] = ((df['won']==1) & df['method'].str.contains('Sub|Submission', case=False, na=False)).astype(float)
    df['_fin'] = ((df['won']==1) & df['method'].str.contains('KO|TKO|Sub', case=False, na=False)).astype(float)
    g = df.groupby('fighter', sort=False)
    df['cum_fights'] = g.cumcount()
    df['_cs_fin'] = g['_fin'].cumsum() - df['_fin']
    df['_cs_won'] = g['won'].cumsum()  - df['won']
    safe_n = df['cum_fights'].clip(lower=1)
    df['career_finish_rate_t'] = np.where(df['cum_fights']>0, df['_cs_fin']/safe_n, 0.0)
    df['career_win_rate_t']    = np.where(df['cum_fights']>0, df['_cs_won']/safe_n, 0.5)

    def _roll(series, w, default):
        return series.shift(1).rolling(w, min_periods=1).mean().fillna(default)

    df['last5_won_t']         = g['won'].transform(lambda x: _roll(x, 5, 0.5))
    df['last5_fin_t']         = g['_fin'].transform(lambda x: _roll(x, 5, 0.0))
    df['win_rate_l5_vs_career'] = df['last5_won_t'] - df['career_win_rate_t']
    df['finish_rate_trend']     = df['last5_fin_t'] - df['career_finish_rate_t']

    def _fights_since_finish(series_fin):
        shifted = series_fin.shift(1)
        result  = []; count = 0
        for v in shifted:
            if pd.isna(v): result.append(0.0)
            else:
                result.append(float(count))
                if v == 1: count = 0
                else:      count += 1
        return pd.Series(result, index=series_fin.index)

    def _max_losing_streak(series_won):
        shifted = series_won.shift(1)
        cur_streak = 0; max_streak = 0; result = []
        for v in shifted:
            if pd.isna(v):
                result.append(max_streak)
            else:
                if v == 0: cur_streak += 1; max_streak = max(max_streak, cur_streak)
                else:      cur_streak = 0
                result.append(max_streak)
        return pd.Series(result, index=series_won.index)

    def _comeback_flag(series_won):
        result = []
        for i in range(len(series_won)):
            window = series_won.shift(1).iloc[max(0,i-10):i].dropna().tolist()
            streak = 0; had_comeback = 0
            for v in window:
                if v == 0: streak += 1
                else:
                    if streak >= 2: had_comeback = 1
                    streak = 0
            result.append(had_comeback)
        return pd.Series(result, index=series_won.index)

    df['fights_since_finish']     = g['_fin'].transform(_fights_since_finish)
    df['longest_lose_streak_ever'] = g['won'].transform(_max_losing_streak)
    df['comeback_flag']            = g['won'].transform(_comeback_flag)

    df.drop(columns=['_ko','_sub','_fin','_cs_fin','_cs_won',
                     'last5_won_t','last5_fin_t',
                     'career_finish_rate_t','career_win_rate_t'], inplace=True)
    return df[['fighter','date','cum_fights',
               'fights_since_finish','win_rate_l5_vs_career',
               'finish_rate_trend','longest_lose_streak_ever','comeback_flag']].copy()

def _layoff_buckets(prefix, days_series):
    d = days_series.fillna(180.0)
    return {
        f'{prefix}layoff_lt90':    (d < 90).astype(int),
        f'{prefix}layoff_90_180':  ((d >= 90)  & (d < 180)).astype(int),
        f'{prefix}layoff_180_365': ((d >= 180) & (d < 365)).astype(int),
        f'{prefix}layoff_gt365':   (d >= 365).astype(int),
    }

def corner_flip(X, y):
    Xf = X.copy()
    for col in list(X.columns):
        if col.startswith('R_'):
            b_col = 'B_' + col[2:]
            if b_col in X.columns:
                Xf[col] = X[b_col].values; Xf[b_col] = X[col].values
    for col in Xf.columns:
        if col.endswith('_dif'): Xf[col] = -Xf[col]
    return pd.concat([X, Xf], ignore_index=True), pd.concat([y, 1-y], ignore_index=True)

def build_full_dataset():
    print('  [D1] Loading data...')
    df = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    career_df = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_df['date'] = pd.to_datetime(career_df['date'])
    career_df = career_df.sort_values(['fighter','date']).reset_index(drop=True)

    style_df = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final_updated.csv'))
    for col in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
        style_df[col] = pd.to_numeric(
            style_df[col].astype(str).str.replace('%','',regex=False), errors='coerce'
        ).fillna(0.0) / 100.0

    print('  [D2] Elo...')
    elo_hist_df = compute_elo(df, K=48, base=1500.0)

    print('  [D3] Career stats...')
    all_win_rates = {f: grp['won'].sum()/max(1,len(grp)) for f,grp in career_df.groupby('fighter')}
    career_stats  = compute_career_stats(career_df, all_win_rates)

    print('  [D4] Trajectory features...')
    traj_stats = compute_trajectory_stats(career_df)

    print('  [D5] Filtering and merging...')
    df = df[df['date'] >= TRAIN_START].copy()
    df = df[df['Winner'].isin(['Red','Blue'])].copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Career stats merge
    career_stats = career_stats.sort_values(['fighter','date'])
    career_cols  = [c for c in career_stats.columns if c not in ('fighter','date')]
    r_career = career_stats.rename(columns={'fighter':'R_fighter', **{c:f'R_{c}' for c in career_cols}})
    b_career = career_stats.rename(columns={'fighter':'B_fighter', **{c:f'B_{c}' for c in career_cols}})
    df = pd.merge_asof(df.sort_values('date'), r_career.sort_values('date'), on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b_career.sort_values('date'), on='date', by='B_fighter', direction='backward')
    career_defaults = {'cum_fights':0,'career_win_rate':0.5,'ko_finish_rate':0.0,'sub_finish_rate':0.0,
                       'career_finish_rate':0.0,'last3_win_rate':0.5,'last10_win_rate':0.5,'last5_won':0.5,
                       'last5_finish_rate':0.0,'trend_score':0.0,'layoff_days':180.0,'opp_quality':0.5}
    for stat, default in career_defaults.items():
        df[f'R_{stat}'] = df[f'R_{stat}'].fillna(default)
        df[f'B_{stat}'] = df[f'B_{stat}'].fillna(default)

    # Elo merge
    elo_cols = elo_hist_df[['fighter','date','elo_before','elo_trend']].copy()
    elo_r = elo_cols.rename(columns={'fighter':'R_fighter','elo_before':'R_elo','elo_trend':'R_elo_trend'})
    elo_b = elo_cols.rename(columns={'fighter':'B_fighter','elo_before':'B_elo','elo_trend':'B_elo_trend'})
    df = pd.merge_asof(df.sort_values('date'), elo_r.sort_values('date'), on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), elo_b.sort_values('date'), on='date', by='B_fighter', direction='backward')
    df['R_elo'] = df['R_elo'].fillna(1500.0); df['B_elo'] = df['B_elo'].fillna(1500.0)
    df['R_elo_trend'] = df['R_elo_trend'].fillna(0.0); df['B_elo_trend'] = df['B_elo_trend'].fillna(0.0)

    # Style merge
    style_src = ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']
    style_df  = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last').reset_index(drop=True)
    r_style   = style_df[['Fighter_Name']+style_src].rename(columns={'Fighter_Name':'R_fighter',**{c:f'R_{c}' for c in style_src}})
    b_style   = style_df[['Fighter_Name']+style_src].rename(columns={'Fighter_Name':'B_fighter',**{c:f'B_{c}' for c in style_src}})
    df = df.merge(r_style, on='R_fighter', how='left')
    df = df.merge(b_style, on='B_fighter', how='left')
    for col in [f'{p}{s}' for p in ('R_','B_') for s in style_src]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Trajectory merge
    traj_stats = traj_stats.sort_values(['fighter','date'])
    traj_cols  = [c for c in traj_stats.columns if c not in ('fighter','date','cum_fights')]
    r_traj = traj_stats.rename(columns={'fighter':'R_fighter', **{c:f'R_{c}' for c in traj_cols}})
    b_traj = traj_stats.rename(columns={'fighter':'B_fighter', **{c:f'B_{c}' for c in traj_cols}})
    df = pd.merge_asof(df.sort_values('date'), r_traj.sort_values('date'), on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b_traj.sort_values('date'), on='date', by='B_fighter', direction='backward')
    traj_defaults = {'fights_since_finish':0.0,'win_rate_l5_vs_career':0.0,
                     'finish_rate_trend':0.0,'longest_lose_streak_ever':0.0,'comeback_flag':0.0}
    for feat, default in traj_defaults.items():
        df[f'R_{feat}'] = df[f'R_{feat}'].fillna(default)
        df[f'B_{feat}'] = df[f'B_{feat}'].fillna(default)

    print('  [D6] Feature engineering...')
    df['weight_class_ord'] = df['weight_class'].map(WC_ORDER).fillna(6).astype(int)
    df['title_bout_bin']   = df['title_bout'].astype(int)
    df['R_southpaw'] = (df['R_Stance'].str.lower() == 'southpaw').astype(int)
    df['B_southpaw'] = (df['B_Stance'].str.lower() == 'southpaw').astype(int)
    df['orth_clash']  = ((df['R_southpaw']==0) & (df['B_southpaw']==0)).astype(int)
    df['south_clash'] = ((df['R_southpaw']==1) & (df['B_southpaw']==1)).astype(int)
    df['R_age'] = pd.to_numeric(df['R_age'], errors='coerce').fillna(28.0)
    df['B_age'] = pd.to_numeric(df['B_age'], errors='coerce').fillna(28.0)
    df['R_age_x_exp'] = df['R_age'] * df['R_cum_fights']
    df['B_age_x_exp'] = df['B_age'] * df['B_cum_fights']
    df['age_x_exp_dif'] = df['R_age_x_exp'] - df['B_age_x_exp']
    for lb in _layoff_buckets('R_', df['R_layoff_days']).items(): df[lb[0]] = lb[1].values
    for lb in _layoff_buckets('B_', df['B_layoff_days']).items(): df[lb[0]] = lb[1].values
    for stat in ['career_win_rate','last5_won','last5_finish_rate','opp_quality','trend_score',
                 'ko_finish_rate','sub_finish_rate','last3_win_rate','last10_win_rate']:
        df[f'{stat}_dif'] = df[f'R_{stat}'] - df[f'B_{stat}']
    df['SLpM_dif']    = df['R_SLpM']    - df['B_SLpM']
    df['SApM_dif']    = df['R_SApM']    - df['B_SApM']
    df['Str_Def_dif'] = df['R_Str_Def'] - df['B_Str_Def']
    df['TD_Def_dif']  = df['R_TD_Def']  - df['B_TD_Def']
    df['Sub_Avg_dif'] = df['R_Sub_Avg'] - df['B_Sub_Avg']
    df['TD_Avg_dif']  = df['R_TD_Avg']  - df['B_TD_Avg']
    df['elo_dif']       = df['R_elo'] - df['B_elo']
    df['elo_trend_dif'] = df['R_elo_trend'] - df['B_elo_trend']
    # Trajectory diffs
    for feat in NEW_TRAJ_FEATS:
        df[f'{feat}_dif'] = df[f'R_{feat}'] - df[f'B_{feat}']

    df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy()
    df['target'] = (df['Winner'] == 'Red').astype(int)
    print(f'  [D7] Dataset ready: {len(df):,} rows')
    return df


def run_variant(name, df, feats, label):
    print(f'\n{"─" * 60}')
    print(f'  VARIANT {name}: {label}')
    print(f'  Features: {len(feats)}')
    print(f'{"─" * 60}')

    # Force numeric
    for col in feats:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    train_mask = df['date'] < TRAIN_CUTOFF
    test_mask  = df['date'] >= TRAIN_CUTOFF

    X_train = df.loc[train_mask, feats].reset_index(drop=True)
    y_train = df.loc[train_mask, 'target'].reset_index(drop=True)
    X_test  = df.loc[test_mask,  feats].reset_index(drop=True)
    y_test  = df.loc[test_mask,  'target'].reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    print(f'  Train (pre-aug): {len(X_train):,}  |  Test: {len(X_test):,}')

    X_aug, y_aug = corner_flip(X_train, y_train)
    print(f'  Train (post-aug): {len(X_aug):,}')

    # LR
    model_lr = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=0.00711, solver='liblinear',
                                   max_iter=2000, random_state=42)),
    ])
    model_lr.fit(X_aug, y_aug)

    # XGB
    model_xgb = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=1)
    model_xgb.fit(X_aug, y_aug)

    # Blend
    prob_lr  = model_lr.predict_proba(X_test)
    prob_xgb = model_xgb.predict_proba(X_test)
    prob     = LR_WEIGHT * prob_lr + XGB_WEIGHT * prob_xgb
    y_pred   = (prob[:, 1] > 0.5).astype(int)
    acc      = accuracy_score(y_test, y_pred)

    print(f'\n  ── Temporal accuracy: {acc:.4f}  ({acc*100:.2f}%) ──')
    df_test['_pred'] = y_pred
    print('  Per-year accuracy:')
    for yr, grp in df_test.groupby(df_test['date'].dt.year):
        yr_acc = accuracy_score(grp['target'], grp['_pred'])
        print(f'    {yr}: {yr_acc:.3f}  ({len(grp):,} fights)')

    # XGB importance top 20
    importances = model_xgb.feature_importances_
    feat_imp = sorted(zip(feats, importances), key=lambda x: -x[1])
    print('\n  Top 10 XGB features:')
    for f, imp in feat_imp[:10]:
        print(f'    {f:<38s}  {imp:.4f}')

    new_in_top = [(f, imp) for f, imp in feat_imp[:20] if any(t in f for t in NEW_TRAJ_FEATS)]
    if new_in_top:
        print('  New traj features in top 20:')
        for f, imp in new_in_top:
            print(f'    {f:<38s}  {imp:.4f}')

    # Save
    var_lr_path  = os.path.join(MODEL, f'variant_{name}_lr.pkl')
    var_xgb_path = os.path.join(MODEL, f'variant_{name}_xgb.pkl')
    var_feats_path = os.path.join(MODEL, f'variant_{name}_features.pkl')
    joblib.dump(model_lr,  var_lr_path)
    joblib.dump(model_xgb, var_xgb_path)
    joblib.dump(feats,     var_feats_path)
    print(f'\n  Saved: variant_{name}_lr.pkl, variant_{name}_xgb.pkl, variant_{name}_features.pkl')

    # Save summary to OUT
    feat_imp_df = pd.DataFrame(feat_imp, columns=['feature','xgb_importance'])
    feat_imp_df.to_csv(os.path.join(OUT, f'variant_{name}_feature_importance.csv'), index=False)

    print(f'\n  {"=" * 55}')
    print(f'  VARIANT {name} SUMMARY')
    print(f'  Label      : {label}')
    print(f'  Features   : {len(feats)}')
    print(f'  Accuracy   : {acc*100:.2f}%  (baseline: 71.47%)')
    delta = (acc - 0.714662) * 100
    print(f'  Delta      : {delta:+.2f} pp vs production model')
    print(f'  {"=" * 55}')

    return acc, feat_imp_df


def main():
    print('=' * 60)
    print('  STEP 4 — Retrain Model 1 (3 Variants)')
    print('=' * 60)

    print('\nBuilding full dataset (includes trajectory features)...')
    df = build_full_dataset()

    FEAT_114 = joblib.load(os.path.join(MODEL, 'feature_columns_best.pkl'))

    # ── Variant A: 114 − removals − redundant ─────────────────────────────────
    feats_A = [f for f in FEAT_114
               if f not in REMOVAL_CANDIDATES and f not in REDUNDANT_DROP]
    print(f'\nVariant A: {len(FEAT_114)} base − {len(REMOVAL_CANDIDATES)} removal − {len(REDUNDANT_DROP)} redundant = {len(feats_A)} features')

    acc_A, imp_A = run_variant('A', df.copy(), feats_A, 'Trimmed (no low-importance + no redundant)')
    gc.collect()

    # ── Variant B: 114 + new trajectory ───────────────────────────────────────
    traj_additions = []
    for feat in NEW_TRAJ_FEATS:
        traj_additions += [f'R_{feat}', f'B_{feat}', f'{feat}_dif']
    feats_B = FEAT_114 + traj_additions
    print(f'\nVariant B: {len(FEAT_114)} base + {len(traj_additions)} trajectory = {len(feats_B)} features')

    acc_B, imp_B = run_variant('B', df.copy(), feats_B, 'Augmented (base + trajectory)')
    gc.collect()

    # ── Variant C: Trimmed A + new trajectory ─────────────────────────────────
    feats_C = feats_A + traj_additions
    print(f'\nVariant C: {len(feats_A)} trimmed + {len(traj_additions)} trajectory = {len(feats_C)} features')

    acc_C, imp_C = run_variant('C', df.copy(), feats_C, 'Trimmed + Trajectory (recommended)')
    gc.collect()

    # ── Final comparison ──────────────────────────────────────────────────────
    baseline = 0.714662
    print('\n' + '=' * 60)
    print('  STEP 4 FINAL COMPARISON')
    print('=' * 60)
    print(f'  {"Variant":<15s}  {"Accuracy":>10s}  {"Delta vs baseline":>20s}')
    print(f'  {"-"*15}  {"-"*10}  {"-"*20}')
    print(f'  {"Baseline (prod)":<15s}  {baseline*100:>9.2f}%  {"—":>20s}')
    for name, acc in [('A (Trimmed)', acc_A), ('B (Augmented)', acc_B), ('C (Trim+Traj)', acc_C)]:
        delta = (acc - baseline) * 100
        print(f'  {name:<15s}  {acc*100:>9.2f}%  {delta:>+19.2f}pp')

    # Save comparison CSV
    comp = pd.DataFrame([
        {'variant': 'Baseline', 'accuracy': round(baseline, 6), 'n_features': 114, 'delta_pp': 0.0},
        {'variant': 'A_Trimmed', 'accuracy': round(acc_A, 6), 'n_features': len(feats_A), 'delta_pp': round((acc_A - baseline)*100, 2)},
        {'variant': 'B_Augmented', 'accuracy': round(acc_B, 6), 'n_features': len(feats_B), 'delta_pp': round((acc_B - baseline)*100, 2)},
        {'variant': 'C_TrimPlusTraj', 'accuracy': round(acc_C, 6), 'n_features': len(feats_C), 'delta_pp': round((acc_C - baseline)*100, 2)},
    ])
    comp.to_csv(os.path.join(OUT, 'variant_comparison.csv'), index=False)
    print(f'\n✓ Comparison saved to {os.path.join(OUT, "variant_comparison.csv")}')
    print('=' * 60)
    print('  STEP 4 COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
