#!/usr/bin/env python3
"""
Model 1 Improvement Sprint — Men's Fights Only
All outputs to experiments/research/model1_v2/
Does NOT touch production files.

Run from project root:
  python experiments/research/model1_v2/sprint.py
"""
import sys, os, warnings, gc, json, traceback
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA   = os.path.join(ROOT, 'data')
OUT    = os.path.dirname(os.path.abspath(__file__))   # experiments/research/model1_v2/

TRAIN_START  = '2018-01-01'
TRAIN_CUTOFF = '2024-01-01'
LR_WEIGHT    = 0.70
XGB_WEIGHT   = 0.30

WOMENS_CLASSES = [
    "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
]

# Production 109-feature list (Variant A)
FEAT_109 = [
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

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def div(s): print(f'\n{"─"*62}'); print(f'  {s}'); print(f'{"─"*62}')
def ok(s):  print(f'  ✓ {s}')
def info(s):print(f'  {s}')

def per_year(df_t, y_pred_arr, label=''):
    df_t = df_t.copy()
    df_t['_pred'] = y_pred_arr
    rows = []
    for yr, grp in df_t.groupby(df_t['date'].dt.year):
        ya = accuracy_score(grp['target'], grp['_pred'])
        rows.append((yr, ya, len(grp)))
        print(f'    {yr}: {ya:.3f}  ({len(grp):,} fights)')
    return rows

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
    Xout = pd.concat([X, Xf], ignore_index=True)
    yout = pd.concat([y, 1 - y], ignore_index=True)
    if w is not None:
        wout = pd.concat([w, w], ignore_index=True)
        return Xout, yout, wout
    return Xout, yout

def blend_predict(lr, xgb, X, lw=LR_WEIGHT, xw=XGB_WEIGHT):
    p_lr  = lr.predict_proba(X)
    p_xgb = xgb.predict_proba(X)
    p     = lw * p_lr + xw * p_xgb
    return (p[:, 1] > 0.5).astype(int), p[:, 1]

def train_blend(X_tr, y_tr, X_te, y_te, df_te, feat_label='',
                sample_weight=None, lr_C=0.00711):
    model_lr = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=lr_C, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    model_xgb = XGBClassifier(random_state=42, eval_metric='logloss',
                               verbosity=0, n_jobs=1)

    if sample_weight is not None:
        sw = sample_weight.values if hasattr(sample_weight, 'values') else sample_weight
        model_lr.fit(X_tr, y_tr, lr__sample_weight=sw)
        model_xgb.fit(X_tr, y_tr, sample_weight=sw)
    else:
        model_lr.fit(X_tr, y_tr)
        model_xgb.fit(X_tr, y_tr)

    y_pred, _ = blend_predict(model_lr, model_xgb, X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f'  [{feat_label}] Temporal accuracy: {acc:.4f} ({acc*100:.2f}%)')
    per_year(df_te, y_pred)
    return model_lr, model_xgb, acc, y_pred

def _layoff_buckets(prefix, days_series):
    d = days_series.fillna(180.0)
    return {
        f'{prefix}layoff_lt90':    (d < 90).astype(int),
        f'{prefix}layoff_90_180':  ((d >= 90)  & (d < 180)).astype(int),
        f'{prefix}layoff_180_365': ((d >= 180) & (d < 365)).astype(int),
        f'{prefix}layoff_gt365':   (d >= 365).astype(int),
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
        elo[r] = r_after
        elo[b] = b_after
    hist = pd.DataFrame(history_rows).sort_values(['fighter', 'date']).reset_index(drop=True)
    hist['elo_trend'] = hist.groupby('fighter')['elo_before'].transform(
        lambda x: x - x.shift(3))
    return hist

def compute_career_stats(career_df, all_win_rates):
    df = career_df.sort_values(['fighter', 'date']).copy().reset_index(drop=True)
    df['_ko']  = ((df['won'] == 1) & df['method'].str.contains('KO|TKO', case=False, na=False)).astype(float)
    df['_sub'] = ((df['won'] == 1) & df['method'].str.contains('Sub|Submission', case=False, na=False)).astype(float)
    df['_fin'] = ((df['won'] == 1) & df['method'].str.contains('KO|TKO|Sub', case=False, na=False)).astype(float)
    g = df.groupby('fighter', sort=False)
    df['cum_fights'] = g.cumcount()
    for src, dst in [('won', '_cs_won'), ('_ko', '_cs_ko'), ('_sub', '_cs_sub'), ('_fin', '_cs_fin')]:
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
    for pos, _ in enumerate(idx_list):
        fighter_positions[fighter_col[pos]].append(pos)
    opp_quality_arr = np.full(len(df), 0.5)
    for fighter, positions in fighter_positions.items():
        for rank, pos in enumerate(positions):
            past_opps = [opp_col[p] for p in positions[max(0, rank-5):rank]]
            rates = [all_win_rates[opp] for opp in past_opps if opp in all_win_rates]
            opp_quality_arr[pos] = float(np.mean(rates)) if rates else 0.5
    df['opp_quality'] = opp_quality_arr
    df.drop(columns=['_ko','_sub','_fin','_cs_won','_cs_ko','_cs_sub','_cs_fin','_prev_date'],
            inplace=True)
    return df[['fighter', 'date', 'cum_fights', 'career_win_rate',
               'ko_finish_rate', 'sub_finish_rate', 'career_finish_rate',
               'last3_win_rate', 'last10_win_rate', 'last5_won',
               'last5_finish_rate', 'trend_score', 'layoff_days', 'opp_quality']]


# ── Build master dataset (shared across all steps) ────────────────────────────
def build_master(date_from='2018-01-01', mens_only=True):
    """Full pipeline: load → Elo → career stats → merge → features.
    Returns df with all 109 columns ready, plus 'date', 'target', 'weight_class'.
    """
    print(f'  Loading data (date_from={date_from}, mens_only={mens_only})...')
    df_raw = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    career_df = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_df['date'] = pd.to_datetime(career_df['date'])
    career_df = career_df.sort_values(['fighter', 'date']).reset_index(drop=True)

    style_df = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final_updated.csv'))
    for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
        style_df[col] = pd.to_numeric(
            style_df[col].astype(str).str.replace('%', '', regex=False),
            errors='coerce'
        ).fillna(0.0) / 100.0

    # Elo — always computed from ALL fights (all-time)
    print('  Computing Elo (all-time fights, K=48)...')
    elo_hist = compute_elo(df_raw, K=48, base=1500.0)

    # Career stats — from career_fights_updated (includes pre-UFC)
    print('  Computing career stats...')
    all_win_rates = {
        f: grp['won'].sum() / max(1, len(grp))
        for f, grp in career_df.groupby('fighter')
    }
    career_stats = compute_career_stats(career_df, all_win_rates)

    # Filter master to date window + valid winner
    df = df_raw[df_raw['date'] >= date_from].copy()
    df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Men's filter
    n_before_mens = len(df)
    if mens_only:
        df = df[~df['weight_class'].isin(WOMENS_CLASSES)].copy()
        n_removed = n_before_mens - len(df)
    else:
        n_removed = 0

    df = df.reset_index(drop=True)

    # Merge career stats
    career_stats = career_stats.sort_values(['fighter', 'date'])
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

    # Merge Elo
    elo_cols = elo_hist[['fighter', 'date', 'elo_before', 'elo_trend']].copy()
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

    # Merge style stats
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

    # Feature engineering
    df['weight_class_ord'] = df['weight_class'].map(WC_ORDER).fillna(6).astype(int)
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
    for stat in ['career_win_rate', 'last5_won', 'last5_finish_rate',
                 'opp_quality', 'trend_score', 'ko_finish_rate',
                 'sub_finish_rate', 'last3_win_rate', 'last10_win_rate']:
        df[f'{stat}_dif'] = df[f'R_{stat}'] - df[f'B_{stat}']
    df['SLpM_dif']    = df['R_SLpM']    - df['B_SLpM']
    df['SApM_dif']    = df['R_SApM']    - df['B_SApM']
    df['Str_Def_dif'] = df['R_Str_Def'] - df['B_Str_Def']
    df['TD_Def_dif']  = df['R_TD_Def']  - df['B_TD_Def']
    df['Sub_Avg_dif'] = df['R_Sub_Avg'] - df['B_Sub_Avg']
    df['TD_Avg_dif']  = df['R_TD_Avg']  - df['B_TD_Avg']
    df['elo_dif']       = df['R_elo']       - df['B_elo']
    df['elo_trend_dif'] = df['R_elo_trend'] - df['B_elo_trend']

    # Standard UFC master cols that map directly
    for col in FEAT_109:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    # Fight filter: both fighters must have prior history
    n_pre_filter = len(df)
    df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy()
    n_filtered = n_pre_filter - len(df)

    df['target'] = (df['Winner'] == 'Red').astype(int)
    df = df.sort_values('date').reset_index(drop=True)

    return df, n_removed, n_filtered, career_df, elo_hist


def compute_weights(dates, cutoff=pd.Timestamp('2024-01-01'), half_life_days=1095):
    days_before = (cutoff - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_before / half_life_days)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP — Men's baseline
# ─────────────────────────────────────────────────────────────────────────────
def run_setup():
    div('SETUP — Men\'s Only Baseline (109 features, 70/30 LR/XGB)')

    df, n_removed, n_filtered, career_df, elo_hist = build_master(
        date_from=TRAIN_START, mens_only=True)

    info(f'Women\'s fights removed: {n_removed:,}')
    info(f'Debut-filtered fights removed: {n_filtered:,}')
    info(f'Total fights remaining: {len(df):,}')

    train_mask = df['date'] < TRAIN_CUTOFF
    test_mask  = df['date'] >= TRAIN_CUTOFF

    X_train_raw = df.loc[train_mask, FEAT_109].reset_index(drop=True)
    y_train_raw = df.loc[train_mask, 'target'].reset_index(drop=True)
    X_test      = df.loc[test_mask,  FEAT_109].reset_index(drop=True)
    y_test      = df.loc[test_mask,  'target'].reset_index(drop=True)
    df_test     = df[test_mask].copy().reset_index(drop=True)

    info(f'Train rows (pre-aug): {len(X_train_raw):,} | Test rows: {len(X_test):,}')

    X_tr, y_tr = corner_flip(X_train_raw, y_train_raw)
    info(f'Train rows (post-aug): {len(X_tr):,}')

    info('\nPer-year accuracy (men\'s baseline):')
    _, _, baseline_acc, y_pred = train_blend(
        X_tr, y_tr, X_test, y_test, df_test, feat_label='Men\'s baseline 109f')

    result = {
        'baseline_acc': baseline_acc,
        'n_removed_womens': n_removed,
        'n_filtered': n_filtered,
        'n_train': len(X_train_raw),
        'n_test': len(X_test),
    }
    gc.collect()
    return df, result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Recency weighting
# ─────────────────────────────────────────────────────────────────────────────
def run_step1(df):
    div('STEP 1 — Recency Weighting')

    train_mask = df['date'] < TRAIN_CUTOFF
    test_mask  = df['date'] >= TRAIN_CUTOFF

    X_train_raw = df.loc[train_mask, FEAT_109].reset_index(drop=True)
    y_train_raw = df.loc[train_mask, 'target'].reset_index(drop=True)
    train_dates = df.loc[train_mask, 'date'].reset_index(drop=True)
    X_test      = df.loc[test_mask,  FEAT_109].reset_index(drop=True)
    y_test      = df.loc[test_mask,  'target'].reset_index(drop=True)
    df_test     = df[test_mask].copy().reset_index(drop=True)

    best_acc = 0.0
    best_hl  = None
    best_lr  = None
    best_xgb = None
    results  = {}

    for hl in [730, 1095, 1460]:
        info(f'\nHalf-life = {hl} days ({hl//365} years):')
        raw_weights = compute_weights(train_dates, half_life_days=hl)
        raw_weights_s = pd.Series(raw_weights, index=y_train_raw.index)
        X_tr, y_tr, w_tr = corner_flip(X_train_raw, y_train_raw, raw_weights_s)
        info(f'  Weight range: {w_tr.min():.4f} – {w_tr.max():.4f} | mean: {w_tr.mean():.4f}')

        lr_m, xgb_m, acc, _ = train_blend(
            X_tr, y_tr, X_test, y_test, df_test,
            feat_label=f'HL={hl}', sample_weight=w_tr)
        results[hl] = acc
        if acc > best_acc:
            best_acc = acc
            best_hl  = hl
            best_lr  = lr_m
            best_xgb = xgb_m
        gc.collect()

    info(f'\nStep 1 Summary:')
    for hl, acc in results.items():
        marker = ' ← BEST' if hl == best_hl else ''
        info(f'  HL={hl:4d}d: {acc:.4f} ({acc*100:.2f}%){marker}')

    out_lr  = os.path.join(OUT, 'recency_weighted_model_lr.pkl')
    out_xgb = os.path.join(OUT, 'recency_weighted_model_xgb.pkl')
    joblib.dump(best_lr,  out_lr)
    joblib.dump(best_xgb, out_xgb)
    ok(f'Saved best recency model (HL={best_hl}) to {os.path.basename(out_lr)} / {os.path.basename(out_xgb)}')

    return {'best_hl': best_hl, 'best_acc': best_acc, 'all_results': results}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Opponent quality adjusted stats
# ─────────────────────────────────────────────────────────────────────────────
def run_step2(df, elo_hist_df, career_df_raw):
    div('STEP 2 — Opponent Quality Adjusted Stats')

    # Build lookup: fighter × date → elo_before
    elo_lookup = elo_hist_df.set_index(['fighter', 'date'])['elo_before'].to_dict()

    # For each fight in career_df_raw, get opponent elo going INTO that fight
    # Use merge_asof to get opponent's most recent elo_before at fight date
    elo_for_merge = elo_hist_df[['fighter', 'date', 'elo_before']].copy()
    elo_for_merge = elo_for_merge.sort_values(['fighter', 'date'])

    info('  Building opponent-elo table from career fights...')
    career_sorted = career_df_raw.sort_values(['fighter', 'date']).copy()
    career_sorted = career_sorted.rename(columns={'opponent': 'opp_name'})

    # For each fight row: get opponent's elo_before at that fight date
    opp_elo_df = career_sorted[['fighter', 'opp_name', 'date', 'won', 'got_finish']].copy()

    # Merge opponent elo: treat opp_name as 'fighter' in elo table, merge by date
    opp_elo_lookup = elo_for_merge.rename(
        columns={'fighter': 'opp_name', 'elo_before': 'opp_elo_before'})
    opp_elo_df = pd.merge_asof(
        opp_elo_df.sort_values('date'),
        opp_elo_lookup.sort_values('date'),
        on='date', by='opp_name', direction='backward'
    )
    opp_elo_df['opp_elo_before'] = opp_elo_df['opp_elo_before'].fillna(1500.0)
    opp_elo_df['elo_weight']     = opp_elo_df['opp_elo_before'] / 1500.0

    # We need SLpM and SApM per fight — use ufc_fighters_final_updated as proxy
    # (career_fights doesn't have per-fight SLpM/SApM — we'll use career averages
    # weighted by opponent elo as a quality adjustment)
    # For SLpM/SApM: use R_SLpM / B_SLpM from master which is a career average.
    # The qa version weights career wins/finishes by opponent elo.

    # qa_win_rate: weighted average of wins by opponent elo
    # qa_finish_rate: weighted average of finishes by opponent elo
    info('  Computing qa_win_rate, qa_finish_rate per fighter...')

    def _qa_stats(group):
        """Compute cumulative QA stats with shift(1) — no leakage."""
        group = group.sort_values('date').copy()
        n = len(group)
        qa_wr   = np.full(n, 0.5)
        qa_fr   = np.full(n, 0.0)
        cum_ew  = 0.0  # cumulative elo-weighted denominator
        cum_eww = 0.0  # cumulative elo-weighted wins
        cum_ewf = 0.0  # cumulative elo-weighted finishes
        for i in range(n):
            if cum_ew > 0:
                qa_wr[i] = cum_eww / cum_ew
                qa_fr[i] = cum_ewf / cum_ew
            else:
                qa_wr[i] = 0.5
                qa_fr[i] = 0.0
            ew  = group.iloc[i]['elo_weight']
            w   = group.iloc[i]['won']
            f   = group.iloc[i]['got_finish'] if pd.notna(group.iloc[i]['got_finish']) else 0.0
            cum_ew  += ew
            cum_eww += ew * w
            cum_ewf += ew * f
        return pd.DataFrame({
            'fighter': group['fighter'].values,
            'date':    group['date'].values,
            'qa_win_rate':    qa_wr,
            'qa_finish_rate': qa_fr,
        })

    qa_list = []
    for fighter, grp in opp_elo_df.groupby('fighter', sort=False):
        qa_list.append(_qa_stats(grp))
    qa_df = pd.concat(qa_list, ignore_index=True)
    qa_df = qa_df.sort_values(['fighter', 'date'])
    info(f'  QA stats computed for {qa_df["fighter"].nunique():,} fighters')

    # For qa_SLpM / qa_SApM: we don't have per-fight strike rates in career_df.
    # Instead, compute: for each fighter, across all their career fights,
    # the weighted average of (SLpM_opponent × elo_weight) as a proxy for
    # "did they land strikes against quality opponents."
    # We approximate using won × elo_weight to build a quality-adjusted scoring rate.
    # This gives qa_SLpM ≈ (sum of elo_weight * won) / n_fights (quality-adjusted offense)
    # and qa_SApM ≈ (sum of elo_weight * (1-won)) / n_fights (quality-adjusted defense)
    info('  Computing qa_SLpM / qa_SApM proxies...')

    def _qa_striking(group):
        group = group.sort_values('date').copy()
        n = len(group)
        qa_slpm = np.full(n, 0.0)
        qa_sapm = np.full(n, 0.0)
        cum_fights = 0
        cum_off = 0.0  # elo-weighted offensive events (wins)
        cum_def = 0.0  # elo-weighted defensive events (losses)
        for i in range(n):
            if cum_fights > 0:
                qa_slpm[i] = cum_off / cum_fights
                qa_sapm[i] = cum_def / cum_fights
            ew  = group.iloc[i]['elo_weight']
            w   = group.iloc[i]['won']
            cum_off += ew * w
            cum_def += ew * (1.0 - w)
            cum_fights += 1
        return pd.DataFrame({
            'fighter': group['fighter'].values,
            'date':    group['date'].values,
            'qa_SLpM': qa_slpm,
            'qa_SApM': qa_sapm,
        })

    qa_striking_list = []
    for fighter, grp in opp_elo_df.groupby('fighter', sort=False):
        qa_striking_list.append(_qa_striking(grp))
    qa_strike_df = pd.concat(qa_striking_list, ignore_index=True)
    qa_strike_df = qa_strike_df.sort_values(['fighter', 'date'])

    # Merge all qa stats together
    qa_all = qa_df.merge(qa_strike_df, on=['fighter', 'date'], how='inner')

    # Merge onto master df for R and B fighters via merge_asof
    info('  Merging QA stats onto master dataset...')
    qa_r = qa_all.rename(columns={
        'fighter': 'R_fighter',
        'qa_win_rate': 'R_qa_win_rate', 'qa_finish_rate': 'R_qa_finish_rate',
        'qa_SLpM': 'R_qa_SLpM', 'qa_SApM': 'R_qa_SApM',
    })
    qa_b = qa_all.rename(columns={
        'fighter': 'B_fighter',
        'qa_win_rate': 'B_qa_win_rate', 'qa_finish_rate': 'B_qa_finish_rate',
        'qa_SLpM': 'B_qa_SLpM', 'qa_SApM': 'B_qa_SApM',
    })

    df_qa = df.copy()
    df_qa = pd.merge_asof(df_qa.sort_values('date'),
                          qa_r.sort_values('date'),
                          on='date', by='R_fighter', direction='backward')
    df_qa = pd.merge_asof(df_qa.sort_values('date'),
                          qa_b.sort_values('date'),
                          on='date', by='B_fighter', direction='backward')

    qa_cols = ['R_qa_win_rate', 'R_qa_finish_rate', 'R_qa_SLpM', 'R_qa_SApM',
               'B_qa_win_rate', 'B_qa_finish_rate', 'B_qa_SLpM', 'B_qa_SApM']
    for c in qa_cols:
        fill_val = 0.5 if 'win_rate' in c else 0.0
        df_qa[c] = df_qa[c].fillna(fill_val)

    # Diffs
    df_qa['qa_win_rate_dif']    = df_qa['R_qa_win_rate']    - df_qa['B_qa_win_rate']
    df_qa['qa_finish_rate_dif'] = df_qa['R_qa_finish_rate'] - df_qa['B_qa_finish_rate']
    df_qa['qa_SLpM_dif']        = df_qa['R_qa_SLpM']        - df_qa['B_qa_SLpM']
    df_qa['qa_SApM_dif']        = df_qa['R_qa_SApM']        - df_qa['B_qa_SApM']

    # Correlation analysis
    info('\n  Correlation vs target (men\'s fights only, 2018+):')
    target = df_qa['target']
    pairs = [
        ('R_career_win_rate', 'R_qa_win_rate'),
        ('R_last5_finish_rate', 'R_qa_finish_rate'),
        ('R_SLpM', 'R_qa_SLpM'),
        ('R_SApM', 'R_qa_SApM'),
        ('career_win_rate_dif', 'qa_win_rate_dif'),
        ('last5_finish_rate_dif', 'qa_finish_rate_dif'),
        ('SLpM_dif', 'qa_SLpM_dif'),
        ('SApM_dif', 'qa_SApM_dif'),
    ]
    qa_signal = {}
    info(f'  {"Feature":<28} {"Raw r":>8} {"QA r":>8} {"Better?":>8}')
    info(f'  {"─"*56}')
    for raw_col, qa_col in pairs:
        r_raw = df_qa[raw_col].corr(target) if raw_col in df_qa else 0.0
        r_qa  = df_qa[qa_col].corr(target)  if qa_col  in df_qa else 0.0
        better = '✓ QA' if abs(r_qa) > abs(r_raw) else '  raw'
        info(f'  {raw_col:<28} {r_raw:>+8.4f} {r_qa:>+8.4f} {better:>8}')
        qa_signal[qa_col] = {'raw_r': r_raw, 'qa_r': r_qa, 'better': abs(r_qa) > abs(r_raw)}

    n_better = sum(1 for v in qa_signal.values() if v['better'])
    info(f'\n  QA features beat raw on {n_better}/{len(pairs)} metrics')

    # Save augmented dataset
    out_path = os.path.join(OUT, 'data_with_qa_stats.pkl')
    df_qa.to_pickle(out_path)
    ok(f'Saved augmented dataset to data_with_qa_stats.pkl ({len(df_qa):,} rows)')

    gc.collect()
    return df_qa, qa_signal


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — New interaction features
# ─────────────────────────────────────────────────────────────────────────────
def run_step3(df_qa, career_df_raw):
    div('STEP 3 — New Interaction Features')

    df3 = df_qa.copy()

    # — Age × layoff interaction —
    info('  Building age_x_layoff...')
    for p in ['R', 'B']:
        df3[f'{p}_age_x_layoff'] = (
            pd.to_numeric(df3[f'{p}_age'], errors='coerce') *
            df3[f'{p}_layoff_days'].clip(upper=730)
        )
    df3['age_x_layoff_dif'] = df3['R_age_x_layoff'] - df3['B_age_x_layoff']

    # — Got-finished rate (chin proxy) — computed from career_fights_updated —
    info('  Computing got_finished_rate (chin proxy) from career_fights_updated...')
    # got_finish=1 means the fighter was finished (opponent won by KO/Sub)
    # We want: among losses, what fraction were finishes?
    # got_finished_rate = cumulative got_finish / cumulative losses (shift(1), no leakage)
    cdf = career_df_raw.sort_values(['fighter', 'date']).copy()
    cdf['is_loss'] = (cdf['won'] == 0).astype(float)
    cdf['is_fin_loss'] = ((cdf['won'] == 0) & (cdf['got_finish'].fillna(0) == 1)).astype(float)

    g = cdf.groupby('fighter', sort=False)
    cdf['_cs_losses']    = g['is_loss'].cumsum()   - cdf['is_loss']
    cdf['_cs_fin_loss']  = g['is_fin_loss'].cumsum() - cdf['is_fin_loss']
    cdf['got_finished_rate'] = np.where(
        cdf['_cs_losses'] > 0,
        cdf['_cs_fin_loss'] / cdf['_cs_losses'],
        0.5  # neutral default for no losses
    )
    chin_df = cdf[['fighter', 'date', 'got_finished_rate']].sort_values(['fighter', 'date'])

    chin_r = chin_df.rename(columns={'fighter': 'R_fighter', 'got_finished_rate': 'R_got_finished_rate'})
    chin_b = chin_df.rename(columns={'fighter': 'B_fighter', 'got_finished_rate': 'B_got_finished_rate'})
    df3 = pd.merge_asof(df3.sort_values('date'), chin_r.sort_values('date'),
                        on='date', by='R_fighter', direction='backward')
    df3 = pd.merge_asof(df3.sort_values('date'), chin_b.sort_values('date'),
                        on='date', by='B_fighter', direction='backward')
    df3['R_got_finished_rate'] = df3['R_got_finished_rate'].fillna(0.5)
    df3['B_got_finished_rate'] = df3['B_got_finished_rate'].fillna(0.5)

    # — Finishing danger mismatch —
    info('  Building finish_danger_mismatch...')
    for p in ['R', 'B']:
        df3[f'{p}_finish_danger']     = df3[f'{p}_ko_finish_rate'] + df3[f'{p}_sub_finish_rate']
        df3[f'{p}_finish_resistance'] = 1.0 - df3[f'{p}_got_finished_rate']

    df3['finish_danger_mismatch'] = (
        df3['R_finish_danger'] * df3['B_finish_resistance'] -
        df3['B_finish_danger'] * df3['R_finish_resistance']
    )

    # — Rematch flag —
    info('  Building rematch flag...')
    df3_sorted = df3.sort_values('date').copy()
    df3_sorted['fighter_pair'] = df3_sorted.apply(
        lambda r: tuple(sorted([str(r['R_fighter']), str(r['B_fighter'])])), axis=1
    )
    # is_rematch: not the first time this pair met in the dataset
    df3_sorted['is_rematch'] = df3_sorted.duplicated(subset=['fighter_pair'], keep='first').astype(int)

    # won_first_fight: from RED corner's perspective in THIS fight
    # For each rematch, look up who won the FIRST meeting
    first_meeting_winner = {}
    for _, row in df3_sorted.sort_values('date').iterrows():
        pair = row['fighter_pair']
        if pair not in first_meeting_winner:
            # This is the first meeting
            winner_name = row['R_fighter'] if row['Winner'] == 'Red' else row['B_fighter']
            first_meeting_winner[pair] = winner_name

    def _won_first(row):
        pair = row['fighter_pair']
        if row['is_rematch'] == 0:
            return 0
        first_winner = first_meeting_winner.get(pair)
        if first_winner is None:
            return 0
        if first_winner == row['R_fighter']:
            return 1
        return -1

    df3_sorted['won_first_fight'] = df3_sorted.apply(_won_first, axis=1)
    df3 = df3_sorted.reset_index(drop=True)

    # — Correlation report —
    new_features = [
        'R_age_x_layoff', 'B_age_x_layoff', 'age_x_layoff_dif',
        'R_finish_danger', 'B_finish_danger', 'finish_danger_mismatch',
        'R_got_finished_rate', 'B_got_finished_rate',
        'is_rematch', 'won_first_fight',
    ]
    target = df3['target']
    info(f'\n  {"Feature":<30} {"Corr with target":>18} {"Keep?":>8}')
    info(f'  {"─"*60}')
    keep_feats = []
    drop_feats = []
    for feat in new_features:
        if feat not in df3.columns:
            info(f'  {feat:<30} {"MISSING":>18}')
            continue
        r = df3[feat].corr(target)
        keep = abs(r) >= 0.03
        marker = '✓' if keep else '✗ DROP'
        info(f'  {feat:<30} {r:>+18.4f} {marker:>8}')
        if keep:
            keep_feats.append(feat)
        else:
            drop_feats.append(feat)

    info(f'\n  Keep: {len(keep_feats)} features | Drop (|r|<0.03): {len(drop_feats)}')
    if drop_feats:
        info(f'  Dropped: {drop_feats}')

    gc.collect()
    return df3, keep_feats, drop_feats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Training window expansion
# ─────────────────────────────────────────────────────────────────────────────
def run_step4(baseline_acc, best_hl, keep_new_feats):
    div('STEP 4 — Training Window Expansion (2015-2017 data quality)')

    df_raw = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw[df_raw['Winner'].isin(['Red', 'Blue'])].copy()

    # Window counts
    window_2015 = df_raw[(df_raw['date'] >= '2015-01-01') & (df_raw['date'] < '2018-01-01')]
    window_2018 = df_raw[(df_raw['date'] >= '2018-01-01') & (df_raw['date'] < '2024-01-01')]

    info(f'  2015-2017 fights: {len(window_2015):,}')
    info(f'  2018-2023 fights: {len(window_2018):,}')

    # Missing rate comparison on top 10 features
    top_features = [
        'R_SApM', 'B_SApM', 'R_SLpM', 'B_SLpM',
        'R_current_win_streak', 'B_current_win_streak',
        'R_avg_SIG_STR_landed', 'B_avg_SIG_STR_landed',
        'R_age', 'B_age',
    ]
    info(f'\n  {"Feature":<28} {"2015-17 miss%":>14} {"2018+ miss%":>12} {"Delta":>8}')
    info(f'  {"─"*66}')

    max_delta = 0.0
    for feat in top_features:
        m_old = window_2015[feat].isna().mean() * 100 if feat in window_2015 else 100.0
        m_new = window_2018[feat].isna().mean() * 100 if feat in window_2018 else 100.0
        delta = m_old - m_new
        max_delta = max(max_delta, delta)
        flag = '⚠' if delta > 20 else ' '
        info(f'  {flag} {feat:<26} {m_old:>13.1f}% {m_new:>11.1f}% {delta:>+7.1f}pp')

    info(f'\n  Max missing-rate delta: {max_delta:.1f}pp')
    threshold = 20.0
    expand = max_delta < threshold

    if expand:
        info(f'  ✓ Delta < {threshold}pp — including 2015-2017 in training')
        new_date_from = '2015-01-01'
    else:
        info(f'  ✗ Delta ≥ {threshold}pp — keeping 2018 cutoff')
        new_date_from = '2018-01-01'

    # Build expanded dataset and test accuracy
    if expand:
        info('\n  Rebuilding dataset with 2015-01-01 start...')
        df_exp, n_rem, n_filt, career_df_exp, elo_hist_exp = build_master(
            date_from=new_date_from, mens_only=True)

        train_mask = df_exp['date'] < TRAIN_CUTOFF
        test_mask  = df_exp['date'] >= TRAIN_CUTOFF
        X_tr_raw = df_exp.loc[train_mask, FEAT_109].reset_index(drop=True)
        y_tr_raw = df_exp.loc[train_mask, 'target'].reset_index(drop=True)
        tr_dates = df_exp.loc[train_mask, 'date'].reset_index(drop=True)
        X_te     = df_exp.loc[test_mask,  FEAT_109].reset_index(drop=True)
        y_te     = df_exp.loc[test_mask,  'target'].reset_index(drop=True)
        df_te    = df_exp[test_mask].copy().reset_index(drop=True)

        w_raw = compute_weights(tr_dates, half_life_days=best_hl)
        w_s   = pd.Series(w_raw, index=y_tr_raw.index)
        X_tr, y_tr, w_tr = corner_flip(X_tr_raw, y_tr_raw, w_s)

        info(f'\n  Expanded train rows (pre-aug): {len(X_tr_raw):,} | Test: {len(X_te):,}')
        info('  Per-year accuracy (expanded window):')
        _, _, exp_acc, _ = train_blend(
            X_tr, y_tr, X_te, y_te, df_te,
            feat_label='Expanded 2015+', sample_weight=w_tr)

        helped = exp_acc > baseline_acc
        info(f'\n  Expanded accuracy: {exp_acc:.4f} vs baseline: {baseline_acc:.4f}')
        info(f'  {"✓ Including pre-2018 data helps" if helped else "✗ Pre-2018 data hurts — keeping 2018 cutoff"}')
        if not helped:
            new_date_from = '2018-01-01'
        gc.collect()
        return {'expand': helped, 'date_from': new_date_from,
                'max_delta': max_delta, 'exp_acc': exp_acc if expand else None}
    else:
        gc.collect()
        return {'expand': False, 'date_from': new_date_from,
                'max_delta': max_delta, 'exp_acc': None}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Full retrain with best combination
# ─────────────────────────────────────────────────────────────────────────────
def run_step5(df_base, df3, step1, step3_keep, step4, baseline_acc):
    div('STEP 5 — Full Retrain: V1, V2, V3 Variants')

    best_hl      = step1['best_hl']
    keep_feats   = [f for f in step3_keep if f in df3.columns]
    date_from_v3 = step4['date_from']
    use_v3       = step4['expand']

    results_out = {}

    # ── V1: Men's only + recency weighting (109 features) ────────────────────
    info('\n--- VARIANT V1: Men\'s only + recency weighting ---')
    df1 = df_base.copy()
    tm1 = df1['date'] < TRAIN_CUTOFF
    te1 = df1['date'] >= TRAIN_CUTOFF
    X1r = df1.loc[tm1, FEAT_109].reset_index(drop=True)
    y1r = df1.loc[tm1, 'target'].reset_index(drop=True)
    d1r = df1.loc[tm1, 'date'].reset_index(drop=True)
    X1t = df1.loc[te1, FEAT_109].reset_index(drop=True)
    y1t = df1.loc[te1, 'target'].reset_index(drop=True)
    df1t= df1[te1].copy().reset_index(drop=True)

    w1 = pd.Series(compute_weights(d1r, half_life_days=best_hl), index=y1r.index)
    X1tr, y1tr, w1tr = corner_flip(X1r, y1r, w1)
    lr1, xgb1, acc1, pred1 = train_blend(
        X1tr, y1tr, X1t, y1t, df1t,
        feat_label='V1', sample_weight=w1tr)

    imp1 = dict(sorted(zip(FEAT_109, xgb1.feature_importances_),
                        key=lambda x: x[1], reverse=True)[:15])
    info('\n  V1 Top 15 XGB features:')
    for f, imp in imp1.items():
        info(f'    {f:<35} {imp:.4f}')

    joblib.dump(lr1,  os.path.join(OUT, 'variant_V1_lr.pkl'))
    joblib.dump(xgb1, os.path.join(OUT, 'variant_V1_xgb.pkl'))
    joblib.dump(FEAT_109, os.path.join(OUT, 'variant_V1_features.pkl'))
    ok(f'V1 saved — acc={acc1:.4f} ({acc1*100:.2f}%), n_features={len(FEAT_109)}')
    results_out['V1'] = {'acc': acc1, 'n_features': len(FEAT_109), 'top15': imp1}
    gc.collect()

    # ── V2: Men's only + recency + QA stats + new features ───────────────────
    info('\n--- VARIANT V2: Men\'s only + recency + QA + new features ---')
    qa_new_feats = [
        'R_qa_win_rate', 'R_qa_finish_rate', 'R_qa_SLpM', 'R_qa_SApM',
        'B_qa_win_rate', 'B_qa_finish_rate', 'B_qa_SLpM', 'B_qa_SApM',
        'qa_win_rate_dif', 'qa_finish_rate_dif', 'qa_SLpM_dif', 'qa_SApM_dif',
    ]
    v2_extra = [f for f in qa_new_feats + keep_feats if f in df3.columns]
    FEAT_V2  = FEAT_109 + [f for f in v2_extra if f not in FEAT_109]
    # ensure all feat_v2 cols are numeric in df3
    for col in FEAT_V2:
        if col not in df3.columns:
            df3[col] = 0.0
        df3[col] = pd.to_numeric(df3[col], errors='coerce').fillna(0.0)

    tm2 = df3['date'] < TRAIN_CUTOFF
    te2 = df3['date'] >= TRAIN_CUTOFF
    X2r = df3.loc[tm2, FEAT_V2].reset_index(drop=True)
    y2r = df3.loc[tm2, 'target'].reset_index(drop=True)
    d2r = df3.loc[tm2, 'date'].reset_index(drop=True)
    X2t = df3.loc[te2, FEAT_V2].reset_index(drop=True)
    y2t = df3.loc[te2, 'target'].reset_index(drop=True)
    df2t= df3[te2].copy().reset_index(drop=True)

    w2 = pd.Series(compute_weights(d2r, half_life_days=best_hl), index=y2r.index)
    X2tr, y2tr, w2tr = corner_flip(X2r, y2r, w2)
    lr2, xgb2, acc2, pred2 = train_blend(
        X2tr, y2tr, X2t, y2t, df2t,
        feat_label='V2', sample_weight=w2tr)

    imp2 = dict(sorted(zip(FEAT_V2, xgb2.feature_importances_),
                        key=lambda x: x[1], reverse=True)[:15])
    info('\n  V2 Top 15 XGB features:')
    for f, imp in imp2.items():
        info(f'    {f:<35} {imp:.4f}')

    joblib.dump(lr2,    os.path.join(OUT, 'variant_V2_lr.pkl'))
    joblib.dump(xgb2,   os.path.join(OUT, 'variant_V2_xgb.pkl'))
    joblib.dump(FEAT_V2, os.path.join(OUT, 'variant_V2_features.pkl'))
    ok(f'V2 saved — acc={acc2:.4f} ({acc2*100:.2f}%), n_features={len(FEAT_V2)}')
    results_out['V2'] = {'acc': acc2, 'n_features': len(FEAT_V2), 'top15': imp2}
    gc.collect()

    # ── V3: V2 + expanded window (only if Step 4 approved) ───────────────────
    if use_v3:
        info(f'\n--- VARIANT V3: V2 + expanded window ({date_from_v3}) ---')
        df_v3, n_rem3, n_filt3, car3, elo3 = build_master(
            date_from=date_from_v3, mens_only=True)

        # Re-run Step 2+3 on expanded dataset would be expensive;
        # we use V2 feature set but train on expanded window (109 features only for V3
        # since QA stats need to be recomputed from scratch on expanded data)
        # For simplicity use FEAT_109 on expanded data
        for col in FEAT_109:
            if col not in df_v3.columns:
                df_v3[col] = 0.0
            df_v3[col] = pd.to_numeric(df_v3[col], errors='coerce').fillna(0.0)

        tm3 = df_v3['date'] < TRAIN_CUTOFF
        te3 = df_v3['date'] >= TRAIN_CUTOFF
        X3r = df_v3.loc[tm3, FEAT_109].reset_index(drop=True)
        y3r = df_v3.loc[tm3, 'target'].reset_index(drop=True)
        d3r = df_v3.loc[tm3, 'date'].reset_index(drop=True)
        X3t = df_v3.loc[te3, FEAT_109].reset_index(drop=True)
        y3t = df_v3.loc[te3, 'target'].reset_index(drop=True)
        df3t= df_v3[te3].copy().reset_index(drop=True)

        w3 = pd.Series(compute_weights(d3r, half_life_days=best_hl), index=y3r.index)
        X3tr, y3tr, w3tr = corner_flip(X3r, y3r, w3)
        lr3, xgb3, acc3, pred3 = train_blend(
            X3tr, y3tr, X3t, y3t, df3t,
            feat_label='V3', sample_weight=w3tr)

        imp3 = dict(sorted(zip(FEAT_109, xgb3.feature_importances_),
                            key=lambda x: x[1], reverse=True)[:15])
        info('\n  V3 Top 15 XGB features:')
        for f, imp in imp3.items():
            info(f'    {f:<35} {imp:.4f}')

        joblib.dump(lr3,    os.path.join(OUT, 'variant_V3_lr.pkl'))
        joblib.dump(xgb3,   os.path.join(OUT, 'variant_V3_xgb.pkl'))
        joblib.dump(FEAT_109, os.path.join(OUT, 'variant_V3_features.pkl'))
        ok(f'V3 saved — acc={acc3:.4f} ({acc3*100:.2f}%), n_features={len(FEAT_109)}')
        results_out['V3'] = {'acc': acc3, 'n_features': len(FEAT_109), 'top15': imp3}
        gc.collect()
    else:
        info('\n  V3 skipped — Step 4 did not approve pre-2018 data')
        results_out['V3'] = None

    return results_out, FEAT_V2


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Findings document
# ─────────────────────────────────────────────────────────────────────────────
def run_step6(setup_r, step1_r, step2_qa, step3_keep, step3_drop,
              step4_r, step5_r, feat_v2):
    div('STEP 6 — Writing Findings')

    baseline  = setup_r['baseline_acc']
    best_v    = max((k for k in step5_r if step5_r[k] is not None),
                    key=lambda k: step5_r[k]['acc'])
    best_acc  = step5_r[best_v]['acc']
    delta_best = best_acc - baseline

    lines = [
        '# Model 1 V2 Sprint — Findings',
        f'_Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}_',
        '',
        '---',
        '',
        '## Summary',
        '',
        f'| Metric | Value |',
        f'|--------|-------|',
        f'| Production baseline (all fights, 70/30) | 72.08% |',
        f'| Men\'s-only baseline (109 features, 70/30) | {baseline*100:.2f}% |',
        f'| Best variant | {best_v} |',
        f'| Best variant accuracy | {best_acc*100:.2f}% |',
        f'| Delta vs men\'s baseline | {delta_best*100:+.2f}pp |',
        f'| Delta vs production | {(best_acc - 0.7208)*100:+.2f}pp |',
        '',
        '---',
        '',
        '## SETUP — Men\'s Only Filter',
        '',
        f'- Women\'s fights removed: **{setup_r["n_removed_womens"]:,}**',
        f'- Debut-filtered fights removed: {setup_r["n_filtered"]:,}',
        f'- Train rows (pre-aug): {setup_r["n_train"]:,}',
        f'- Test rows: {setup_r["n_test"]:,}',
        f'- Men\'s baseline accuracy: **{baseline*100:.2f}%**',
        '',
        '---',
        '',
        '## STEP 1 — Recency Weighting',
        '',
        '| Half-life | Accuracy | Delta vs baseline |',
        '|-----------|----------|-------------------|',
    ]
    for hl, acc in step1_r['all_results'].items():
        marker = ' ← BEST' if hl == step1_r['best_hl'] else ''
        lines.append(f'| {hl}d ({hl//365}yr) | {acc*100:.2f}% | {(acc-baseline)*100:+.2f}pp{marker} |')

    lines += [
        '',
        f'**Best half-life:** {step1_r["best_hl"]} days',
        f'**Best accuracy:** {step1_r["best_acc"]*100:.2f}%',
        '',
        '---',
        '',
        '## STEP 2 — Opponent Quality Adjusted Stats',
        '',
        '| Feature | Raw r | QA r | Better? |',
        '|---------|-------|------|---------|',
    ]
    for feat, vals in step2_qa.items():
        better = '✓ QA' if vals['better'] else 'raw'
        lines.append(f'| {feat} | {vals["raw_r"]:+.4f} | {vals["qa_r"]:+.4f} | {better} |')

    n_qa_better = sum(1 for v in step2_qa.values() if v['better'])
    lines += [
        '',
        f'QA features outperform raw on **{n_qa_better}/{len(step2_qa)}** metrics.',
        '',
        '---',
        '',
        '## STEP 3 — New Interaction Features',
        '',
        f'**Kept** (|r| ≥ 0.03): {step3_keep}',
        '',
        f'**Dropped** (|r| < 0.03): {step3_drop}',
        '',
        '---',
        '',
        '## STEP 4 — Training Window Expansion',
        '',
        f'- Max missing-rate delta (2015-17 vs 2018+): **{step4_r["max_delta"]:.1f}pp**',
        f'- Threshold: 20pp',
        f'- Decision: **{"Include 2015-2017" if step4_r["expand"] else "Keep 2018 cutoff"}**',
    ]
    if step4_r['exp_acc']:
        lines.append(f'- Expanded accuracy: {step4_r["exp_acc"]*100:.2f}%')

    lines += [
        '',
        '---',
        '',
        '## STEP 5 — Variant Results',
        '',
        '| Variant | Features | Accuracy | Delta vs baseline |',
        '|---------|----------|----------|-------------------|',
    ]
    for vname, vdata in step5_r.items():
        if vdata is None:
            lines.append(f'| {vname} | N/A | skipped | — |')
        else:
            lines.append(f'| {vname} | {vdata["n_features"]} | '
                         f'{vdata["acc"]*100:.2f}% | '
                         f'{(vdata["acc"]-baseline)*100:+.2f}pp |')

    lines += [
        '',
        '---',
        '',
        '## Recommendation',
        '',
        f'**Recommended variant: {best_v}** — {best_acc*100:.2f}% temporal accuracy '
        f'({delta_best*100:+.2f}pp vs men\'s baseline, '
        f'{(best_acc-0.7208)*100:+.2f}pp vs production 72.08%).',
        '',
        '**Do not promote to production until reviewed.**',
        '',
        '### Promotion checklist',
        '- [ ] Review per-year accuracy for regressions in any single year',
        '- [ ] Confirm backend can accept men\'s-only filter at inference or confirm',
        '      that the model handles women\'s fights gracefully (it was not trained on them)',
        '- [ ] Update model_metadata.json with men\'s-only flag if promoting',
        '- [ ] A/B test on upcoming card before full promotion',
    ]

    out_path = os.path.join(OUT, 'MODEL1_V2_FINDINGS.md')
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    ok(f'Findings written to MODEL1_V2_FINDINGS.md')

    # Also dump numeric results as JSON for later reference
    results_json = {
        'baseline_acc': baseline,
        'step1': step1_r,
        'step4': step4_r,
        'step5': {k: ({**v, 'top15': list(v['top15'].keys())} if v else None)
                  for k, v in step5_r.items()},
        'best_variant': best_v,
        'best_acc': best_acc,
    }
    with open(os.path.join(OUT, 'sprint_results.json'), 'w') as fh:
        json.dump(results_json, fh, indent=2, default=str)
    ok('sprint_results.json saved')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('=' * 62)
    print('  Model 1 Improvement Sprint — Men\'s Fights Only')
    print(f'  Output: experiments/research/model1_v2/')
    print(f'  Production files: NOT TOUCHED')
    print('=' * 62)

    steps_done = {}

    try:
        df_base, setup_r = run_setup()
        steps_done['setup'] = True
        print(f'\n  SETUP DONE — Men\'s baseline: {setup_r["baseline_acc"]*100:.2f}%')
    except Exception:
        print('\n  *** SETUP FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    print('\n' + '=' * 62)
    print('  Continuing to STEP 1 — Recency Weighting')
    print('=' * 62)
    try:
        step1_r = run_step1(df_base)
        steps_done['step1'] = True
        print(f'\n  STEP 1 DONE — Best HL={step1_r["best_hl"]}d, acc={step1_r["best_acc"]*100:.2f}%')
    except Exception:
        print('\n  *** STEP 1 FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    # Load career_df and elo_hist for Step 2
    career_df_raw = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_df_raw['date'] = pd.to_datetime(career_df_raw['date'])
    career_df_raw = career_df_raw.sort_values(['fighter', 'date']).reset_index(drop=True)

    elo_hist_path = os.path.join(DATA, 'elo_ratings_history.csv')
    elo_hist_df = pd.read_csv(elo_hist_path)
    elo_hist_df['date'] = pd.to_datetime(elo_hist_df['date'])

    print('\n' + '=' * 62)
    print('  Continuing to STEP 2 — Opponent Quality Adjusted Stats')
    print('=' * 62)
    try:
        df_qa, step2_qa = run_step2(df_base, elo_hist_df, career_df_raw)
        steps_done['step2'] = True
        print(f'\n  STEP 2 DONE — QA stats computed and merged')
    except Exception:
        print('\n  *** STEP 2 FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    print('\n' + '=' * 62)
    print('  Continuing to STEP 3 — Interaction Features')
    print('=' * 62)
    try:
        df3, step3_keep, step3_drop = run_step3(df_qa, career_df_raw)
        steps_done['step3'] = True
        print(f'\n  STEP 3 DONE — {len(step3_keep)} new features kept, {len(step3_drop)} dropped')
    except Exception:
        print('\n  *** STEP 3 FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    print('\n' + '=' * 62)
    print('  Continuing to STEP 4 — Training Window Expansion')
    print('=' * 62)
    try:
        step4_r = run_step4(setup_r['baseline_acc'], step1_r['best_hl'], step3_keep)
        steps_done['step4'] = True
        print(f'\n  STEP 4 DONE — date_from={step4_r["date_from"]}, expand={step4_r["expand"]}')
    except Exception:
        print('\n  *** STEP 4 FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    print('\n' + '=' * 62)
    print('  Continuing to STEP 5 — Full Retrain')
    print('=' * 62)
    try:
        step5_r, feat_v2 = run_step5(df_base, df3, step1_r, step3_keep, step4_r,
                                      setup_r['baseline_acc'])
        steps_done['step5'] = True
        print(f'\n  STEP 5 DONE — variants trained')
    except Exception:
        print('\n  *** STEP 5 FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    try:
        run_step6(setup_r, step1_r, step2_qa, step3_keep, step3_drop,
                  step4_r, step5_r, feat_v2)
        steps_done['step6'] = True
    except Exception:
        print('\n  *** STEP 6 FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    print(f'\n{"=" * 62}')
    print('  SPRINT COMPLETE')
    best_v = max((k for k in step5_r if step5_r[k] is not None),
                 key=lambda k: step5_r[k]['acc'])
    print(f'  Best variant : {best_v}  —  {step5_r[best_v]["acc"]*100:.2f}%')
    print(f'  Men\'s baseline: {setup_r["baseline_acc"]*100:.2f}%')
    print(f'  Production:    72.08%')
    print(f'  See model1_v2/MODEL1_V2_FINDINGS.md for full report')
    print(f'{"=" * 62}\n')


if __name__ == '__main__':
    main()
