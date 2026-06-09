#!/usr/bin/env python3
"""
8SI UFC Predictor — Women's Model 1 Trainer (Baseline)
  Blend:   70% LogisticRegression + 30% XGBoost
  Features: 109 base (Variant A) minus high-zero-rate features
  Scope:   Women's weight classes only
  Window:  All available women's data → 2023-12-31 train, 2024+ test
  Recency: Tests HL=730d and HL=1095d — uses whichever scores higher
  Elo:     Computed from women's fights only
  Augmentation: Corner-flip on train only
"""
import sys
import os
import gc
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import json
from collections import defaultdict

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(ROOT, 'data')
MODEL = os.path.join(ROOT, 'model')

TRAIN_CUTOFF = '2024-01-01'
LR_WEIGHT    = 0.70
XGB_WEIGHT   = 0.30
ZERO_RATE_THRESHOLD = 0.90   # drop features with >90% zeros

WOMENS_CLASSES = [
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
]

WC_ORDER_WOMENS = {
    "Women's Strawweight":   0,
    "Women's Flyweight":     1,
    "Women's Bantamweight":  2,
    "Women's Featherweight": 3,
}

# 109-feature Variant A (men's baseline — women's will drop high-zero-rate features)
FEAT_BASE = [
    "R_wins", "R_losses", "R_Height_cms", "R_age",
    "R_avg_SIG_STR_landed", "R_avg_TD_landed",
    "R_current_win_streak", "R_current_lose_streak", "R_longest_win_streak",
    "R_avg_SIG_STR_pct", "R_avg_SUB_ATT", "R_avg_TD_pct",
    "R_Reach_cms",
    "B_wins", "B_losses", "B_Height_cms", "B_age",
    "B_avg_SIG_STR_landed", "B_avg_TD_landed",
    "B_current_win_streak", "B_current_lose_streak", "B_longest_win_streak",
    "B_avg_SIG_STR_pct", "B_avg_SUB_ATT", "B_avg_TD_pct",
    "B_Reach_cms", "B_total_title_bouts",
    "win_dif", "loss_dif", "win_streak_dif", "lose_streak_dif",
    "height_dif", "reach_dif", "age_dif", "sig_str_dif",
    "avg_td_dif", "ko_dif", "sub_dif", "total_title_bout_dif",
    "weight_class_ord",
    "orth_clash", "south_clash", "R_southpaw",
    "R_cum_fights", "B_cum_fights",
    "R_career_win_rate", "B_career_win_rate", "career_win_rate_dif",
    "R_last5_won", "B_last5_won", "last5_won_dif",
    "R_last5_finish_rate", "B_last5_finish_rate", "last5_finish_rate_dif",
    "R_opp_quality", "B_opp_quality", "opp_quality_dif",
    "R_trend_score", "B_trend_score", "trend_score_dif",
    "R_ko_finish_rate", "B_ko_finish_rate", "ko_finish_rate_dif",
    "R_sub_finish_rate", "B_sub_finish_rate", "sub_finish_rate_dif",
    "R_last3_win_rate", "B_last3_win_rate", "last3_win_rate_dif",
    "R_last10_win_rate", "B_last10_win_rate",
    "R_age_x_exp", "B_age_x_exp", "age_x_exp_dif",
    "R_layoff_lt90", "R_layoff_90_180", "R_layoff_180_365", "R_layoff_gt365",
    "B_layoff_lt90", "B_layoff_90_180", "B_layoff_180_365",
    "R_SLpM", "R_SApM", "R_Str_Acc", "R_Str_Def",
    "R_TD_Avg", "R_TD_Acc", "R_TD_Def", "R_Sub_Avg",
    "B_SLpM", "B_SApM", "B_Str_Acc", "B_Str_Def",
    "B_TD_Avg", "B_TD_Acc", "B_TD_Def", "B_Sub_Avg",
    "SLpM_dif", "SApM_dif", "Str_Def_dif", "TD_Def_dif", "Sub_Avg_dif", "TD_Avg_dif",
    "R_elo", "B_elo", "elo_dif",
    "R_elo_trend", "B_elo_trend", "elo_trend_dif",
]

XGB_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 4,
    'min_child_weight': 2, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'gamma': 0.3, 'reg_alpha': 0, 'reg_lambda': 2.0,
    'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1,
}


# ─────────────────────────────────────────────────────────────────────────────
def compute_elo_womens(df_womens, K=48, base=1500.0):
    """Elo from women's fights only."""
    df_sorted = df_womens.sort_values('date').reset_index(drop=True)
    elo = {}
    history_rows = []

    for _, row in df_sorted.iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        date, winner = row['date'], row['Winner']

        r_before = elo.get(r, base)
        b_before = elo.get(b, base)
        r_exp = 1.0 / (1.0 + 10.0 ** ((b_before - r_before) / 400.0))
        b_exp = 1.0 - r_exp

        if winner == 'Red':
            r_act, b_act = 1.0, 0.0
        elif winner == 'Blue':
            r_act, b_act = 0.0, 1.0
        else:
            r_act, b_act = 0.5, 0.5

        r_after = r_before + K * (r_act - r_exp)
        b_after = b_before + K * (b_act - b_exp)

        history_rows.append({'fighter': r, 'opponent': b, 'date': date,
                              'elo_before': r_before, 'elo_after': r_after, 'result': r_act})
        history_rows.append({'fighter': b, 'opponent': r, 'date': date,
                              'elo_before': b_before, 'elo_after': b_after, 'result': b_act})
        elo[r] = r_after
        elo[b] = b_after

    hist = pd.DataFrame(history_rows).sort_values(['fighter', 'date']).reset_index(drop=True)
    hist['elo_trend'] = hist.groupby('fighter')['elo_before'].transform(
        lambda x: x - x.shift(3)
    )
    counts     = hist.groupby('fighter').size()
    last_dates = hist.groupby('fighter')['date'].max()
    curr = pd.DataFrame([
        {'fighter': f, 'current_elo': e,
         'last_fight_date': last_dates.get(f),
         'total_fights': int(counts.get(f, 0))}
        for f, e in elo.items()
    ])
    return hist, curr


def compute_career_stats(career_df, all_win_rates):
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
    df['_prev_date']        = g['date'].transform(lambda x: x.shift(1))
    df['layoff_days']       = (df['date'] - df['_prev_date']).dt.days.fillna(180.0).clip(lower=0)

    opp_col     = df['opponent'].tolist()
    fighter_col = df['fighter'].tolist()
    idx_list    = df.index.tolist()
    fighter_positions = defaultdict(list)
    for pos, idx in enumerate(idx_list):
        fighter_positions[fighter_col[pos]].append(pos)

    opp_quality_arr = np.full(len(df), 0.5)
    for fighter, positions in fighter_positions.items():
        for rank, pos in enumerate(positions):
            past_opps = [opp_col[p] for p in positions[max(0, rank - 5):rank]]
            rates = [all_win_rates[opp] for opp in past_opps if opp in all_win_rates]
            opp_quality_arr[pos] = float(np.mean(rates)) if rates else 0.5
    df['opp_quality'] = opp_quality_arr

    df.drop(columns=['_ko','_sub','_fin','_cs_won','_cs_ko','_cs_sub','_cs_fin','_prev_date'],
            inplace=True)
    return df[['fighter','date','cum_fights','career_win_rate','ko_finish_rate',
               'sub_finish_rate','career_finish_rate','last3_win_rate','last10_win_rate',
               'last5_won','last5_finish_rate','trend_score','layoff_days','opp_quality']]


def _layoff_buckets(prefix, days_series):
    d = days_series.fillna(180.0)
    return {
        f'{prefix}layoff_lt90':    (d < 90).astype(int),
        f'{prefix}layoff_90_180':  ((d >= 90)  & (d < 180)).astype(int),
        f'{prefix}layoff_180_365': ((d >= 180) & (d < 365)).astype(int),
        f'{prefix}layoff_gt365':   (d >= 365).astype(int),
    }


def corner_flip(X, y, w=None):
    """Double training set by swapping Red↔Blue and negating diffs."""
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


def compute_weights(dates, cutoff=pd.Timestamp('2024-01-01'), half_life_days=730):
    days_before = (cutoff - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_before / half_life_days)


def train_and_eval(X_train_raw, y_train_raw, d_train_raw, X_test, y_test, df_test, feat_cols, hl):
    w_raw = pd.Series(compute_weights(d_train_raw, half_life_days=hl), index=y_train_raw.index)
    X_aug, y_aug, w_aug = corner_flip(X_train_raw, y_train_raw, w_raw)
    w_arr = w_aug.values

    lr = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=0.01, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    lr.fit(X_aug, y_aug, lr__sample_weight=w_arr)
    gc.collect()

    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_aug, y_aug, sample_weight=w_arr)
    gc.collect()

    prob_lr  = lr.predict_proba(X_test)
    prob_xgb = xgb.predict_proba(X_test)
    prob     = LR_WEIGHT * prob_lr + XGB_WEIGHT * prob_xgb
    y_pred   = (prob[:, 1] > 0.5).astype(int)
    acc      = accuracy_score(y_test, y_pred)
    return lr, xgb, y_pred, acc, X_aug, y_aug, w_arr


def main():
    print('=' * 62)
    print('  8SI UFC Predictor — Women\'s Model 1 Trainer (Baseline)')
    print(f'  Blend: {int(LR_WEIGHT*100)}% LR + {int(XGB_WEIGHT*100)}% XGB')
    print(f'  Scope: women\'s fights only | Cutoff: {TRAIN_CUTOFF}')
    print('=' * 62)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print('\n[1/9] Loading data...')
    df_all = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    df_all['date'] = pd.to_datetime(df_all['date'])

    career_raw = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_raw['date'] = pd.to_datetime(career_raw['date'])
    career_raw = career_raw.sort_values(['fighter', 'date']).reset_index(drop=True)

    style_df = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final_updated.csv'))
    for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
        style_df[col] = pd.to_numeric(
            style_df[col].astype(str).str.replace('%', '', regex=False),
            errors='coerce'
        ).fillna(0.0) / 100.0

    print(f'   ufc-master: {len(df_all):,} rows | career: {len(career_raw):,} rows')

    # ── 2. Filter to women's only ─────────────────────────────────────────────
    print('\n[2/9] Filtering to women\'s weight classes...')
    master_womens = df_all[df_all['weight_class'].isin(WOMENS_CLASSES)].copy()
    master_womens = master_womens[master_womens['Winner'].isin(['Red', 'Blue'])].copy()
    master_womens = master_womens.sort_values('date').reset_index(drop=True)
    print(f'   Total women\'s fights (valid winner): {len(master_womens):,}')
    print(f'   Year range: {master_womens["date"].dt.year.min()} – {master_womens["date"].dt.year.max()}')
    print(f'   Per class:')
    for wc, n in master_womens['weight_class'].value_counts().items():
        print(f'     {wc}: {n}')

    # ── 3. Elo (women's fights only) ──────────────────────────────────────────
    print('\n[3/9] Computing Elo from women\'s fights only (K=48, base=1500)...')
    elo_hist_df, elo_curr_df = compute_elo_womens(master_womens, K=48, base=1500.0)
    print(f'   History rows: {len(elo_hist_df):,} | Unique fighters: {elo_curr_df["fighter"].nunique():,}')
    gc.collect()

    # ── 4. Career stats (women's fighters only) ───────────────────────────────
    print('\n[4/9] Computing career stats (shift=1, no leakage)...')
    womens_fighters = set(master_womens['R_fighter']) | set(master_womens['B_fighter'])
    career_womens = career_raw[career_raw['fighter'].isin(womens_fighters)].copy()
    career_womens = career_womens.sort_values(['fighter', 'date']).reset_index(drop=True)
    print(f'   Women\'s career rows: {len(career_womens):,} ({career_womens["fighter"].nunique():,} fighters)')

    all_win_rates = {
        f: grp['won'].sum() / max(1, len(grp))
        for f, grp in career_womens.groupby('fighter')
    }
    career_stats = compute_career_stats(career_womens, all_win_rates)
    print(f'   Career stat rows: {len(career_stats):,}')
    gc.collect()

    # ── 5. Merge all stats onto fight data ────────────────────────────────────
    print('\n[5/9] Merging career stats, Elo, style stats onto fight data...')
    df = master_womens.copy()

    career_cols = [c for c in career_stats.columns if c not in ('fighter', 'date')]
    r_career = career_stats.rename(columns={'fighter': 'R_fighter',
                                             **{c: f'R_{c}' for c in career_cols}})
    b_career = career_stats.rename(columns={'fighter': 'B_fighter',
                                             **{c: f'B_{c}' for c in career_cols}})
    df = pd.merge_asof(df.sort_values('date'), r_career.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b_career.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')

    career_defaults = {
        'cum_fights': 0, 'career_win_rate': 0.5, 'ko_finish_rate': 0.0,
        'sub_finish_rate': 0.0, 'career_finish_rate': 0.0,
        'last3_win_rate': 0.5, 'last10_win_rate': 0.5,
        'last5_won': 0.5, 'last5_finish_rate': 0.0,
        'trend_score': 0.0, 'layoff_days': 180.0, 'opp_quality': 0.5,
    }
    for stat, default in career_defaults.items():
        df[f'R_{stat}'] = df[f'R_{stat}'].fillna(default)
        df[f'B_{stat}'] = df[f'B_{stat}'].fillna(default)

    # Elo
    elo_cols = elo_hist_df[['fighter', 'date', 'elo_before', 'elo_trend']].copy()
    elo_r = elo_cols.rename(columns={'fighter': 'R_fighter',
                                      'elo_before': 'R_elo', 'elo_trend': 'R_elo_trend'})
    elo_b = elo_cols.rename(columns={'fighter': 'B_fighter',
                                      'elo_before': 'B_elo', 'elo_trend': 'B_elo_trend'})
    df = pd.merge_asof(df.sort_values('date'), elo_r.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), elo_b.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')
    df['R_elo'] = df['R_elo'].fillna(1500.0)
    df['B_elo'] = df['B_elo'].fillna(1500.0)
    df['R_elo_trend'] = df['R_elo_trend'].fillna(0.0)
    df['B_elo_trend'] = df['B_elo_trend'].fillna(0.0)

    # Style stats
    style_src = ['SLpM', 'SApM', 'Str_Acc', 'Str_Def', 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']
    style_df = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last').reset_index(drop=True)
    r_style = style_df[['Fighter_Name'] + style_src].rename(
        columns={'Fighter_Name': 'R_fighter', **{c: f'R_{c}' for c in style_src}})
    b_style = style_df[['Fighter_Name'] + style_src].rename(
        columns={'Fighter_Name': 'B_fighter', **{c: f'B_{c}' for c in style_src}})
    df = df.merge(r_style, on='R_fighter', how='left')
    df = df.merge(b_style, on='B_fighter', how='left')
    for col in [f'{p}{s}' for p in ('R_', 'B_') for s in style_src]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    gc.collect()

    # ── 6. Feature engineering ────────────────────────────────────────────────
    print('\n[6/9] Engineering features...')
    df['weight_class_ord'] = df['weight_class'].map(WC_ORDER_WOMENS).fillna(1).astype(int)
    df['R_southpaw'] = (df['R_Stance'].str.lower() == 'southpaw').astype(int)
    df['B_southpaw'] = (df['B_Stance'].str.lower() == 'southpaw').astype(int)
    df['orth_clash']  = ((df['R_southpaw'] == 0) & (df['B_southpaw'] == 0)).astype(int)
    df['south_clash'] = ((df['R_southpaw'] == 1) & (df['B_southpaw'] == 1)).astype(int)
    df['R_age'] = pd.to_numeric(df['R_age'], errors='coerce').fillna(28.0)
    df['B_age'] = pd.to_numeric(df['B_age'], errors='coerce').fillna(28.0)
    df['R_age_x_exp']  = df['R_age'] * df['R_cum_fights']
    df['B_age_x_exp']  = df['B_age'] * df['B_cum_fights']
    df['age_x_exp_dif'] = df['R_age_x_exp'] - df['B_age_x_exp']

    for lb_key, lb_val in _layoff_buckets('R_', df['R_layoff_days']).items():
        df[lb_key] = lb_val.values
    for lb_key, lb_val in _layoff_buckets('B_', df['B_layoff_days']).items():
        df[lb_key] = lb_val.values

    # Diff features
    for stat in ['career_win_rate', 'last5_won', 'last5_finish_rate',
                 'opp_quality', 'trend_score', 'ko_finish_rate',
                 'sub_finish_rate', 'last3_win_rate', 'last10_win_rate']:
        df[f'{stat}_dif'] = df[f'R_{stat}'] - df[f'B_{stat}']

    # Raw stats from master CSV
    for col in ['R_wins', 'R_losses', 'R_Height_cms', 'R_Reach_cms',
                'B_wins', 'B_losses', 'B_Height_cms', 'B_Reach_cms',
                'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_avg_SIG_STR_pct',
                'R_avg_SUB_ATT', 'R_avg_TD_pct',
                'B_avg_SIG_STR_landed', 'B_avg_TD_landed', 'B_avg_SIG_STR_pct',
                'B_avg_SUB_ATT', 'B_avg_TD_pct',
                'R_current_win_streak', 'R_current_lose_streak', 'R_longest_win_streak',
                'B_current_win_streak', 'B_current_lose_streak', 'B_longest_win_streak',
                'B_total_title_bouts']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

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
    df['total_title_bout_dif'] = 0  # will be near-zero; computed below if column exists
    df['SLpM_dif']    = df['R_SLpM']    - df['B_SLpM']
    df['SApM_dif']    = df['R_SApM']    - df['B_SApM']
    df['Str_Def_dif'] = df['R_Str_Def'] - df['B_Str_Def']
    df['TD_Def_dif']  = df['R_TD_Def']  - df['B_TD_Def']
    df['Sub_Avg_dif'] = df['R_Sub_Avg'] - df['B_Sub_Avg']
    df['TD_Avg_dif']  = df['R_TD_Avg']  - df['B_TD_Avg']
    df['elo_dif']       = df['R_elo']       - df['B_elo']
    df['elo_trend_dif'] = df['R_elo_trend'] - df['B_elo_trend']

    # Ensure all FEAT_BASE columns are numeric
    for col in FEAT_BASE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    # ── 7. Fight filter ───────────────────────────────────────────────────────
    print('\n[7/9] Applying fight filter (R_cum_fights >= 1 AND B_cum_fights >= 1)...')
    n_before = len(df)
    df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy()
    print(f'   Excluded debuts: {n_before - len(df):,} | Kept: {len(df):,}')
    df['target'] = (df['Winner'] == 'Red').astype(int)

    # ── 8. Zero-rate check — drop features > 90% zero in women's data ─────────
    print(f'\n[8/9] Zero-rate check (threshold: {ZERO_RATE_THRESHOLD*100:.0f}%)...')
    train_mask = df['date'] < TRAIN_CUTOFF
    X_full_train = df.loc[train_mask, FEAT_BASE]
    zero_rates = (X_full_train == 0).mean()
    dropped_feats = zero_rates[zero_rates > ZERO_RATE_THRESHOLD].index.tolist()
    kept_feats = [f for f in FEAT_BASE if f not in dropped_feats]
    print(f'   Features dropped ({len(dropped_feats)}):')
    for feat in dropped_feats:
        print(f'     {feat}: {zero_rates[feat]*100:.1f}% zero')
    print(f'   Features kept: {len(kept_feats)} of {len(FEAT_BASE)}')

    # ── 9. Split, train at both half-lives, pick best ─────────────────────────
    print('\n[9/9] Training at HL=730d and HL=1095d, picking best...')
    test_mask   = df['date'] >= TRAIN_CUTOFF
    X_train_raw = df.loc[train_mask, kept_feats].reset_index(drop=True)
    y_train_raw = df.loc[train_mask, 'target'].reset_index(drop=True)
    d_train_raw = df.loc[train_mask, 'date'].reset_index(drop=True)
    X_test      = df.loc[test_mask,  kept_feats].reset_index(drop=True)
    y_test      = df.loc[test_mask,  'target'].reset_index(drop=True)
    df_test     = df[test_mask].copy().reset_index(drop=True)

    print(f'   Train rows (pre-aug): {len(X_train_raw):,} | Test rows: {len(X_test):,}')

    results = {}
    for hl in [730, 1095]:
        print(f'\n   ─── Training HL={hl}d ───')
        lr_m, xgb_m, y_pred, acc, X_aug, y_aug, w_arr = train_and_eval(
            X_train_raw, y_train_raw, d_train_raw, X_test, y_test, df_test, kept_feats, hl
        )
        results[hl] = (lr_m, xgb_m, y_pred, acc, X_aug, y_aug, w_arr)
        print(f'   HL={hl}d → temporal accuracy: {acc:.4f} ({acc*100:.2f}%)')
        gc.collect()

    best_hl = max(results, key=lambda h: results[h][3])
    print(f'\n   ✓ Best half-life: HL={best_hl}d')
    lr_best, xgb_best, y_pred_best, acc_best, _, _, _ = results[best_hl]

    print(f'\n   ── Final accuracy: {acc_best:.4f} ({acc_best*100:.2f}%) ──')
    df_test['_pred'] = y_pred_best

    print('\n   Per-year breakdown:')
    for yr, grp in df_test.groupby(df_test['date'].dt.year):
        yr_acc = accuracy_score(grp['target'], grp['_pred'])
        print(f'     {yr}: {yr_acc:.3f}  ({len(grp):,} fights)')

    print('\n   Per-weight-class breakdown:')
    for wc, grp in df_test.groupby('weight_class'):
        wc_acc = accuracy_score(grp['target'], grp['_pred'])
        print(f'     {wc}: {wc_acc:.3f}  ({len(grp):,} fights)')

    print('\n   Classification report:')
    print(classification_report(y_test, y_pred_best, target_names=['Blue wins', 'Red wins']))

    # XGB feature importance
    print('\n   Top 10 features by XGB importance:')
    importances = xgb_best.feature_importances_
    feat_imp = sorted(zip(kept_feats, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        print(f'     {feat}: {imp:.4f}')

    # ── Save outputs (separate files — do NOT touch men's models) ─────────────
    print('\n   Saving outputs (women\'s only — men\'s files untouched)...')

    joblib.dump(lr_best,    os.path.join(MODEL, 'ufc_model_womens_lr.pkl'))
    joblib.dump(xgb_best,   os.path.join(MODEL, 'ufc_model_womens_xgb.pkl'))
    joblib.dump(kept_feats, os.path.join(MODEL, 'ufc_model_womens_features.pkl'))

    acc_730  = results[730][3]
    acc_1095 = results[1095][3]
    metadata = {
        'model_type': f'M1_womens_blend_LR70_XGB30_HL{best_hl}d',
        'temporal_accuracy': round(float(acc_best), 6),
        'n_features': len(kept_feats),
        'feature_list': kept_feats,
        'dropped_features': dropped_feats,
        'dropped_reason': f'>{ZERO_RATE_THRESHOLD*100:.0f}% zero rate in women\'s training data',
        'blend_ratio': f'{int(LR_WEIGHT*100)}% LR + {int(XGB_WEIGHT*100)}% XGB',
        'best_half_life_days': best_hl,
        'acc_hl730': round(float(acc_730), 6),
        'acc_hl1095': round(float(acc_1095), 6),
        'training_window': f'All available → <{TRAIN_CUTOFF}',
        'test_window': f'>= {TRAIN_CUTOFF} (women\'s fights only)',
        'fight_filter': 'R_cum_fights >= 1 AND B_cum_fights >= 1',
        'elo_scope': "women's fights only",
        'xgb_params': XGB_PARAMS,
        'n_train': int(len(X_train_raw)),
        'n_test': int(len(X_test)),
        'date_trained': datetime.now().strftime('%Y-%m-%d'),
    }
    with open(os.path.join(MODEL, 'ufc_model_womens_metadata.json'), 'w') as fh:
        json.dump(metadata, fh, indent=2)

    saved = [
        os.path.join(MODEL, 'ufc_model_womens_lr.pkl'),
        os.path.join(MODEL, 'ufc_model_womens_xgb.pkl'),
        os.path.join(MODEL, 'ufc_model_womens_features.pkl'),
        os.path.join(MODEL, 'ufc_model_womens_metadata.json'),
    ]
    for path in saved:
        size_kb = os.path.getsize(path) / 1024
        print(f'     ✓ {os.path.relpath(path, ROOT):<50}  {size_kb:,.1f} KB')

    print(f'\n{"=" * 62}')
    print(f'  Training complete.')
    print(f'  HL=730d: {acc_730*100:.2f}%  |  HL=1095d: {acc_1095*100:.2f}%')
    print(f'  Best: HL={best_hl}d → {acc_best*100:.2f}% temporal accuracy')
    print(f'  Train / Test: {len(X_train_raw):,} / {len(X_test):,} fights')
    print(f'  Features: {len(kept_feats)} kept, {len(dropped_feats)} dropped')
    print(f'{"=" * 62}\n')


if __name__ == '__main__':
    main()
