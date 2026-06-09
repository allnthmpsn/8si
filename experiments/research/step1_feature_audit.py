#!/usr/bin/env python3
"""
Step 1 — Feature Audit on Model 1
Computes: correlation with target, XGBoost importance, missing/zero rate.
Flags candidates for removal and redundant feature groups.
Saves: experiments/research/feature_audit.csv
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA  = os.path.join(ROOT, 'data')
MODEL = os.path.join(ROOT, 'model')
OUT   = os.path.join(ROOT, 'experiments', 'research')

TRAIN_START  = '2018-01-01'
TRAIN_CUTOFF = '2024-01-01'

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

def compute_elo(df_all, K=48, base=1500.0):
    df_sorted = df_all.sort_values('date').reset_index(drop=True)
    elo = {}
    history_rows = []
    for _, row in df_sorted.iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        date, winner = row['date'], row['Winner']
        r_before = elo.get(r, base)
        b_before = elo.get(b, base)
        r_exp = 1.0 / (1.0 + 10.0 ** ((b_before - r_before) / 400.0))
        b_exp = 1.0 - r_exp
        if winner == 'Red':   r_act, b_act = 1.0, 0.0
        elif winner == 'Blue': r_act, b_act = 0.0, 1.0
        else:                  r_act, b_act = 0.5, 0.5
        r_after = r_before + K * (r_act - r_exp)
        b_after = b_before + K * (b_act - b_exp)
        history_rows.append({'fighter': r, 'opponent': b, 'date': date,
                              'elo_before': r_before, 'elo_after': r_after, 'result': r_act})
        history_rows.append({'fighter': b, 'opponent': r, 'date': date,
                              'elo_before': b_before, 'elo_after': b_after, 'result': b_act})
        elo[r] = r_after; elo[b] = b_after
    hist = pd.DataFrame(history_rows).sort_values(['fighter', 'date']).reset_index(drop=True)
    hist['elo_trend'] = hist.groupby('fighter')['elo_before'].transform(lambda x: x - x.shift(3))
    return hist

def compute_career_stats(career_df, all_win_rates):
    from collections import defaultdict
    df = career_df.sort_values(['fighter', 'date']).copy().reset_index(drop=True)
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
    opp_col = df['opponent'].tolist(); fighter_col = df['fighter'].tolist(); idx_list = df.index.tolist()
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
    df.drop(columns=['_ko','_sub','_fin','_cs_won','_cs_ko','_cs_sub','_cs_fin','_prev_date'], inplace=True)
    return df[['fighter','date','cum_fights','career_win_rate','ko_finish_rate','sub_finish_rate',
               'career_finish_rate','last3_win_rate','last10_win_rate','last5_won',
               'last5_finish_rate','trend_score','layoff_days','opp_quality']]

def _layoff_buckets(prefix, days_series):
    d = days_series.fillna(180.0)
    return {
        f'{prefix}layoff_lt90':    (d < 90).astype(int),
        f'{prefix}layoff_90_180':  ((d >= 90)  & (d < 180)).astype(int),
        f'{prefix}layoff_180_365': ((d >= 180) & (d < 365)).astype(int),
        f'{prefix}layoff_gt365':   (d >= 365).astype(int),
    }

def build_dataset():
    print('[1] Loading data...')
    df = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    career_df = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_df['date'] = pd.to_datetime(career_df['date'])
    career_df = career_df.sort_values(['fighter', 'date']).reset_index(drop=True)

    style_df = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final_updated.csv'))
    for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
        style_df[col] = pd.to_numeric(
            style_df[col].astype(str).str.replace('%', '', regex=False), errors='coerce'
        ).fillna(0.0) / 100.0

    print('[2] Computing Elo...')
    elo_hist_df = compute_elo(df, K=48, base=1500.0)

    print('[3] Computing career stats...')
    all_win_rates = {f: grp['won'].sum()/max(1,len(grp)) for f,grp in career_df.groupby('fighter')}
    career_stats  = compute_career_stats(career_df, all_win_rates)

    print('[4] Filtering and merging...')
    df = df[df['date'] >= TRAIN_START].copy()
    df = df[df['Winner'].isin(['Red','Blue'])].copy()
    df = df.sort_values('date').reset_index(drop=True)

    career_stats = career_stats.sort_values(['fighter','date'])
    career_cols  = [c for c in career_stats.columns if c not in ('fighter','date')]

    r_career = career_stats.rename(columns={'fighter':'R_fighter', **{c: f'R_{c}' for c in career_cols}})
    b_career = career_stats.rename(columns={'fighter':'B_fighter', **{c: f'B_{c}' for c in career_cols}})

    df = pd.merge_asof(df.sort_values('date'), r_career.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b_career.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')

    career_defaults = {
        'cum_fights':0,'career_win_rate':0.5,'ko_finish_rate':0.0,'sub_finish_rate':0.0,
        'career_finish_rate':0.0,'last3_win_rate':0.5,'last10_win_rate':0.5,'last5_won':0.5,
        'last5_finish_rate':0.0,'trend_score':0.0,'layoff_days':180.0,'opp_quality':0.5,
    }
    for stat, default in career_defaults.items():
        df[f'R_{stat}'] = df[f'R_{stat}'].fillna(default)
        df[f'B_{stat}'] = df[f'B_{stat}'].fillna(default)

    elo_cols = elo_hist_df[['fighter','date','elo_before','elo_trend']].copy()
    elo_r = elo_cols.rename(columns={'fighter':'R_fighter','elo_before':'R_elo','elo_trend':'R_elo_trend'})
    elo_b = elo_cols.rename(columns={'fighter':'B_fighter','elo_before':'B_elo','elo_trend':'B_elo_trend'})
    df = pd.merge_asof(df.sort_values('date'), elo_r.sort_values('date'), on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), elo_b.sort_values('date'), on='date', by='B_fighter', direction='backward')
    df['R_elo'] = df['R_elo'].fillna(1500.0); df['B_elo'] = df['B_elo'].fillna(1500.0)
    df['R_elo_trend'] = df['R_elo_trend'].fillna(0.0); df['B_elo_trend'] = df['B_elo_trend'].fillna(0.0)

    style_src = ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']
    style_df = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last').reset_index(drop=True)
    r_style = style_df[['Fighter_Name']+style_src].rename(columns={'Fighter_Name':'R_fighter',**{c:f'R_{c}' for c in style_src}})
    b_style = style_df[['Fighter_Name']+style_src].rename(columns={'Fighter_Name':'B_fighter',**{c:f'B_{c}' for c in style_src}})
    df = df.merge(r_style, on='R_fighter', how='left')
    df = df.merge(b_style, on='B_fighter', how='left')
    for col in [f'{p}{s}' for p in ('R_','B_') for s in style_src]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    print('[5] Engineering features...')
    df['weight_class_ord'] = df['weight_class'].map(WC_ORDER).fillna(6).astype(int)
    df['title_bout_bin']   = df['title_bout'].astype(int)
    df['R_southpaw'] = (df['R_Stance'].str.lower() == 'southpaw').astype(int)
    df['B_southpaw'] = (df['B_Stance'].str.lower() == 'southpaw').astype(int)
    df['orth_clash']  = ((df['R_southpaw'] == 0) & (df['B_southpaw'] == 0)).astype(int)
    df['south_clash'] = ((df['R_southpaw'] == 1) & (df['B_southpaw'] == 1)).astype(int)
    df['R_age'] = pd.to_numeric(df['R_age'], errors='coerce').fillna(28.0)
    df['B_age'] = pd.to_numeric(df['B_age'], errors='coerce').fillna(28.0)
    df['R_age_x_exp']  = df['R_age'] * df['R_cum_fights']
    df['B_age_x_exp']  = df['B_age'] * df['B_cum_fights']
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

    FEAT_114 = joblib.load(os.path.join(MODEL, 'feature_columns_best.pkl'))
    for col in FEAT_114:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy()
    df['target'] = (df['Winner'] == 'Red').astype(int)
    print(f'   Dataset ready: {len(df):,} rows')
    return df, FEAT_114

def main():
    print('=' * 60)
    print('  STEP 1 — Feature Audit on Model 1')
    print('=' * 60)

    df, FEAT_114 = build_dataset()

    train_mask = df['date'] < TRAIN_CUTOFF
    X_train = df.loc[train_mask, FEAT_114]
    y_train = df.loc[train_mask, 'target']
    X_all   = df[FEAT_114]
    y_all   = df['target']

    print('\n[6] Training XGBoost on train set for importance...')
    xgb = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=1)
    xgb.fit(X_train, y_train)
    importances = xgb.feature_importances_

    print('[7] Computing correlations and missing rates...')
    rows = []
    for i, feat in enumerate(FEAT_114):
        corr   = float(X_all[feat].corr(y_all))
        imp    = float(importances[i])
        n_zero = int((X_all[feat] == 0).sum())
        n_miss = int(X_all[feat].isna().sum())  # should be 0 after fillna
        zero_rate = n_zero / len(X_all)
        rows.append({
            'feature': feat,
            'xgb_importance': round(imp, 6),
            'corr_with_target': round(corr, 4),
            'zero_count': n_zero,
            'zero_rate': round(zero_rate, 4),
            'na_count': n_miss,
        })

    audit = pd.DataFrame(rows).sort_values('xgb_importance', ascending=False).reset_index(drop=True)
    audit['rank'] = audit.index + 1

    # Flag low importance + high zero rate
    imp_thresh  = 0.005   # bottom < 0.5% importance
    zero_thresh = 0.70    # > 70% zero values
    audit['flag_low_importance'] = audit['xgb_importance'] < imp_thresh
    audit['flag_high_zero']      = audit['zero_rate'] > zero_thresh
    audit['flag_removal_candidate'] = audit['flag_low_importance'] & audit['flag_high_zero']

    # Redundancy check — correlation matrix on diff features
    diff_feats = [f for f in FEAT_114 if f.endswith('_dif')]
    corr_matrix = X_all[diff_feats].corr().abs()
    redundant_pairs = []
    for a in diff_feats:
        for b in diff_feats:
            if a < b:
                c = corr_matrix.loc[a, b]
                if c > 0.85:
                    redundant_pairs.append((a, b, round(float(c), 3)))

    out_path = os.path.join(OUT, 'feature_audit.csv')
    audit.to_csv(out_path, index=False)
    print(f'\n   Saved: {out_path}')

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  STEP 1 RESULTS')
    print('=' * 60)

    print(f'\nTop 15 features by XGBoost importance:')
    for _, r in audit.head(15).iterrows():
        print(f"  {r['rank']:3d}. {r['feature']:<30s}  imp={r['xgb_importance']:.4f}  corr={r['corr_with_target']:+.3f}  zero%={r['zero_rate']*100:.1f}%")

    print(f'\nBottom 15 features by XGBoost importance:')
    for _, r in audit.tail(15).iterrows():
        print(f"  {r['rank']:3d}. {r['feature']:<30s}  imp={r['xgb_importance']:.4f}  corr={r['corr_with_target']:+.3f}  zero%={r['zero_rate']*100:.1f}%")

    cands = audit[audit['flag_removal_candidate']]
    print(f'\n⚑  Removal candidates (low imp + high zero): {len(cands)}')
    for _, r in cands.iterrows():
        print(f"     {r['feature']:<30s}  imp={r['xgb_importance']:.4f}  zero%={r['zero_rate']*100:.1f}%")

    print(f'\n⚑  Highly correlated diff-feature pairs (|r| > 0.85): {len(redundant_pairs)}')
    for a, b, c in redundant_pairs[:20]:
        print(f"     {a}  ↔  {b}  ({c})")

    print(f'\n✓ Feature audit saved to {out_path}')
    print('=' * 60)
    print('  STEP 1 COMPLETE')
    print('=' * 60)
    return audit, df, FEAT_114

if __name__ == '__main__':
    main()
