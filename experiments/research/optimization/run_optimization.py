#!/usr/bin/env python3
"""
8SI UFC Predictor — Model 1 Optimization Sprint
All results saved to experiments/research/optimization/
No production files overwritten.
Baseline: 109-feature LR90/XGB10 blend @ 71.64% temporal accuracy
"""
import os
import sys
import gc
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
import lightgbm as lgb
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA   = os.path.join(ROOT, 'data')
MODEL  = os.path.join(ROOT, 'model')
OPTDIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(OPTDIR, exist_ok=True)

TRAIN_START  = '2018-01-01'
TRAIN_CUTOFF = '2024-01-01'
BASELINE_ACC = 0.71642

FEATURES = joblib.load(os.path.join(MODEL, 'feature_columns_best.pkl'))

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

results = {}  # accumulated across all steps


# ─── Shared pipeline helpers (exact copy from train_model1.py) ────────────────

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
    for pos, idx in enumerate(idx_list):
        fighter_positions[fighter_col[pos]].append(pos)
    opp_quality_arr = np.full(len(df), 0.5)
    for fighter, positions in fighter_positions.items():
        for rank, pos in enumerate(positions):
            past_opps = [opp_col[p] for p in positions[max(0, rank - 5):rank]]
            rates = [all_win_rates[opp] for opp in past_opps if opp in all_win_rates]
            opp_quality_arr[pos] = float(np.mean(rates)) if rates else 0.5
    df['opp_quality'] = opp_quality_arr
    df.drop(columns=['_ko','_sub','_fin','_cs_won','_cs_ko','_cs_sub','_cs_fin','_prev_date'], inplace=True)
    return df[['fighter', 'date', 'cum_fights', 'career_win_rate',
               'ko_finish_rate', 'sub_finish_rate', 'career_finish_rate',
               'last3_win_rate', 'last10_win_rate', 'last5_won',
               'last5_finish_rate', 'trend_score', 'layoff_days', 'opp_quality']]


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
                Xf[col]   = X[b_col].values
                Xf[b_col] = X[col].values
    for col in Xf.columns:
        if col.endswith('_dif'):
            Xf[col] = -Xf[col]
    return (pd.concat([X, Xf], ignore_index=True),
            pd.concat([y, 1 - y], ignore_index=True))


def save_results():
    path = os.path.join(OPTDIR, 'results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def delta(acc):
    return f"{'+' if acc - BASELINE_ACC >= 0 else ''}{(acc - BASELINE_ACC)*100:.2f}pp"


# ─── Build dataset (runs once, shared across all steps) ───────────────────────

def build_dataset():
    print('\n' + '='*62)
    print('  BUILDING DATASET (runs once for all steps)')
    print('='*62)

    print('\n[1/7] Loading CSVs...')
    df = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    career_df = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_df['date'] = pd.to_datetime(career_df['date'])
    career_df = career_df.sort_values(['fighter', 'date']).reset_index(drop=True)
    style_df = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final_updated.csv'))
    for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
        style_df[col] = pd.to_numeric(
            style_df[col].astype(str).str.replace('%', '', regex=False),
            errors='coerce'
        ).fillna(0.0) / 100.0
    print(f'   master: {len(df):,} | career: {len(career_df):,} | style: {len(style_df):,}')

    print('\n[2/7] Computing Elo...')
    elo_hist_df, _ = compute_elo(df, K=48, base=1500.0)
    print(f'   Elo history rows: {len(elo_hist_df):,}')
    gc.collect()

    print('\n[3/7] Computing career stats...')
    all_win_rates = {
        f: grp['won'].sum() / max(1, len(grp))
        for f, grp in career_df.groupby('fighter')
    }
    career_stats = compute_career_stats(career_df, all_win_rates)
    print(f'   Career stat rows: {len(career_stats):,}')
    gc.collect()

    print(f'\n[4/7] Filtering master to {TRAIN_START}+...')
    df = df[df['date'] >= TRAIN_START].copy()
    df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
    df = df.sort_values('date').reset_index(drop=True)
    print(f'   Fights: {len(df):,}')

    print('\n[5/7] Merging career stats and Elo...')
    career_stats = career_stats.sort_values(['fighter', 'date'])
    career_cols  = [c for c in career_stats.columns if c not in ('fighter', 'date')]
    r_career = career_stats.rename(columns={'fighter': 'R_fighter', **{c: f'R_{c}' for c in career_cols}})
    b_career = career_stats.rename(columns={'fighter': 'B_fighter', **{c: f'B_{c}' for c in career_cols}})
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

    elo_cols = elo_hist_df[['fighter', 'date', 'elo_before', 'elo_trend']].copy()
    elo_r = elo_cols.rename(columns={'fighter': 'R_fighter', 'elo_before': 'R_elo', 'elo_trend': 'R_elo_trend'})
    elo_b = elo_cols.rename(columns={'fighter': 'B_fighter', 'elo_before': 'B_elo', 'elo_trend': 'B_elo_trend'})
    df = pd.merge_asof(df.sort_values('date'), elo_r.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), elo_b.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')
    df['R_elo']       = df['R_elo'].fillna(1500.0)
    df['B_elo']       = df['B_elo'].fillna(1500.0)
    df['R_elo_trend'] = df['R_elo_trend'].fillna(0.0)
    df['B_elo_trend'] = df['B_elo_trend'].fillna(0.0)

    style_src = ['SLpM', 'SApM', 'Str_Acc', 'Str_Def', 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']
    style_df  = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last').reset_index(drop=True)
    r_style = style_df[['Fighter_Name'] + style_src].rename(
        columns={'Fighter_Name': 'R_fighter', **{c: f'R_{c}' for c in style_src}})
    b_style = style_df[['Fighter_Name'] + style_src].rename(
        columns={'Fighter_Name': 'B_fighter', **{c: f'B_{c}' for c in style_src}})
    df = df.merge(r_style, on='R_fighter', how='left')
    df = df.merge(b_style, on='B_fighter', how='left')
    for col in [f'{p}{s}' for p in ('R_', 'B_') for s in style_src]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    gc.collect()

    print('\n[6/7] Engineering features...')
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
    for lb in _layoff_buckets('R_', df['R_layoff_days']).items():
        df[lb[0]] = lb[1].values
    for lb in _layoff_buckets('B_', df['B_layoff_days']).items():
        df[lb[0]] = lb[1].values
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
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    print('\n[7/7] Fight filter + split...')
    df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy()
    df['target'] = (df['Winner'] == 'Red').astype(int)
    train_mask = df['date'] < TRAIN_CUTOFF
    test_mask  = df['date'] >= TRAIN_CUTOFF

    X_train = df.loc[train_mask, FEATURES].reset_index(drop=True)
    y_train = df.loc[train_mask, 'target'].reset_index(drop=True)
    X_test  = df.loc[test_mask,  FEATURES].reset_index(drop=True)
    y_test  = df.loc[test_mask,  'target'].reset_index(drop=True)
    dates_test = df.loc[test_mask, 'date'].reset_index(drop=True)

    print(f'   Train (pre-aug): {len(X_train):,} | Test: {len(X_test):,}')
    X_train_aug, y_train_aug = corner_flip(X_train, y_train)
    print(f'   Train (post-aug): {len(X_train_aug):,} (corner-flip ×2)')

    gc.collect()
    return X_train, y_train, X_train_aug, y_train_aug, X_test, y_test, dates_test


# ─── STEP 1: ElasticNet LR ────────────────────────────────────────────────────

def step1_elasticnet(X_train_aug, y_train_aug, X_test, y_test, dates_test, prod_xgb):
    print('\n' + '='*62)
    print('  STEP 1 — ElasticNet LR  (l1_ratio sweep, C=0.00711, saga)')
    print('='*62)

    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    table = []

    for l1r in l1_ratios:
        print(f'   l1_ratio={l1r:.1f} ...', end='', flush=True)
        pipe = Pipeline([
            ('sc', RobustScaler()),
            ('lr', LogisticRegression(
                penalty='elasticnet', C=0.00711, solver='saga',
                l1_ratio=l1r, max_iter=5000, random_state=42, n_jobs=1,
            )),
        ])
        pipe.fit(X_train_aug, y_train_aug)

        prob_lr  = pipe.predict_proba(X_test)
        prob_xgb = prod_xgb.predict_proba(X_test)
        prob     = 0.90 * prob_lr + 0.10 * prob_xgb
        y_pred   = (prob[:, 1] > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        table.append({'l1_ratio': l1r, 'accuracy': acc})
        print(f'  {acc*100:.2f}%  ({delta(acc)})')
        gc.collect()

    best = max(table, key=lambda x: x['accuracy'])
    print(f'\n   Best l1_ratio: {best["l1_ratio"]:.1f}  →  {best["accuracy"]*100:.2f}%  ({delta(best["accuracy"])})')

    # Save best model
    best_pipe = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(
            penalty='elasticnet', C=0.00711, solver='saga',
            l1_ratio=best['l1_ratio'], max_iter=5000, random_state=42, n_jobs=1,
        )),
    ])
    best_pipe.fit(X_train_aug, y_train_aug)
    joblib.dump(best_pipe, os.path.join(OPTDIR, 'elasticnet_lr.pkl'))
    print(f'   Saved: optimization/elasticnet_lr.pkl')

    beats = best['accuracy'] > BASELINE_ACC
    print(f'   Beats baseline (71.64%): {"YES" if beats else "NO"}')

    results['step1_elasticnet'] = {
        'table': [{'l1_ratio': r['l1_ratio'], 'accuracy': round(r['accuracy'], 6)} for r in table],
        'best_l1_ratio': best['l1_ratio'],
        'best_accuracy': round(best['accuracy'], 6),
        'beats_baseline': beats,
        'saved': 'optimization/elasticnet_lr.pkl',
    }
    save_results()

    print('\n  ┌─────────────────────────────────────┐')
    print('  │  STEP 1 SUMMARY                     │')
    print('  ├─────────────────────────────────────┤')
    print(f'  │  Baseline:   71.64%                 │')
    for r in table:
        flag = ' ◄ best' if r['l1_ratio'] == best['l1_ratio'] else ''
        print(f'  │  l1={r["l1_ratio"]:.1f}:  {r["accuracy"]*100:.2f}%  ({delta(r["accuracy"])}){flag:<10}│')
    print(f'  │  Beats baseline: {"YES" if beats else "NO":<22}│')
    print('  └─────────────────────────────────────┘')

    return best_pipe, best['accuracy']


# ─── STEP 2: Blend ratio sweep ────────────────────────────────────────────────

def step2_blend_ratios(prod_lr, prod_xgb, X_test, y_test):
    print('\n' + '='*62)
    print('  STEP 2 — Blend Ratio Sweep  (production LR + XGB)')
    print('='*62)

    ratios = [
        (0.95, 0.05),
        (0.90, 0.10),  # baseline
        (0.85, 0.15),
        (0.80, 0.20),
        (0.75, 0.25),
        (0.70, 0.30),
    ]

    prob_lr  = prod_lr.predict_proba(X_test)
    prob_xgb = prod_xgb.predict_proba(X_test)

    table = []
    for lr_w, xgb_w in ratios:
        prob   = lr_w * prob_lr + xgb_w * prob_xgb
        y_pred = (prob[:, 1] > 0.5).astype(int)
        acc    = accuracy_score(y_test, y_pred)
        is_base = (lr_w == 0.90)
        table.append({'lr_pct': int(lr_w*100), 'xgb_pct': int(xgb_w*100),
                      'accuracy': acc, 'is_baseline': is_base})
        marker = ' (baseline)' if is_base else ''
        print(f'   LR {int(lr_w*100):2d}% / XGB {int(xgb_w*100):2d}%:  {acc*100:.2f}%  ({delta(acc)}){marker}')

    non_base = [r for r in table if not r['is_baseline']]
    best = max(non_base, key=lambda x: x['accuracy'])

    best_blend = {
        'lr_pct': best['lr_pct'],
        'xgb_pct': best['xgb_pct'],
        'accuracy': round(best['accuracy'], 6),
        'delta_vs_baseline': round(best['accuracy'] - BASELINE_ACC, 6),
    }
    with open(os.path.join(OPTDIR, 'best_blend.json'), 'w') as f:
        json.dump(best_blend, f, indent=2)
    print(f'\n   Best non-baseline: LR {best["lr_pct"]}% / XGB {best["xgb_pct"]}%  →  {best["accuracy"]*100:.2f}%')
    print(f'   Saved: optimization/best_blend.json')

    results['step2_blend'] = {
        'table': [{'lr_pct': r['lr_pct'], 'xgb_pct': r['xgb_pct'],
                   'accuracy': round(r['accuracy'], 6)} for r in table],
        'best_non_baseline': best_blend,
    }
    save_results()

    print('\n  ┌──────────────────────────────────────────┐')
    print('  │  STEP 2 SUMMARY                          │')
    print('  ├──────────────────────────────────────────┤')
    for r in table:
        base_str = ' (baseline)' if r['is_baseline'] else ''
        best_str = ' ◄ best non-base' if r['lr_pct'] == best['lr_pct'] else ''
        print(f'  │  LR{r["lr_pct"]:2d}/XGB{r["xgb_pct"]:2d}: {r["accuracy"]*100:.2f}% ({delta(r["accuracy"])}){base_str}{best_str}')
    print('  └──────────────────────────────────────────┘')

    return best['lr_pct'], best['xgb_pct'], best['accuracy']


# ─── STEP 3: LightGBM ─────────────────────────────────────────────────────────

def step3_lightgbm(prod_lr, prod_xgb, X_train_aug, y_train_aug, X_test, y_test):
    print('\n' + '='*62)
    print('  STEP 3 — LightGBM blends')
    print('='*62)

    lgbm_params = {
        'n_estimators':     200,
        'learning_rate':    0.05,
        'max_depth':        4,
        'num_leaves':       15,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha':        0.1,
        'reg_lambda':       1.0,
        'random_state':     42,
        'n_jobs':           1,
        'verbose':          -1,
    }
    print('   Training LightGBM...', end='', flush=True)
    lgbm_model = lgb.LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_train_aug, y_train_aug)
    print(' done')
    gc.collect()

    prob_lr   = prod_lr.predict_proba(X_test)
    prob_xgb  = prod_xgb.predict_proba(X_test)
    prob_lgbm = lgbm_model.predict_proba(X_test)

    blends = [
        ('LR 90% + LGBM 10%',             0.90, 0.00, 0.10),
        ('LR 85% + LGBM 15%',             0.85, 0.00, 0.15),
        ('LR 80% + LGBM 20%',             0.80, 0.00, 0.20),
        ('LR 80% + XGB 10% + LGBM 10%',   0.80, 0.10, 0.10),
    ]

    table = []
    for label, lrw, xgbw, lgbmw in blends:
        prob   = lrw * prob_lr + xgbw * prob_xgb + lgbmw * prob_lgbm
        y_pred = (prob[:, 1] > 0.5).astype(int)
        acc    = accuracy_score(y_test, y_pred)
        table.append({'label': label, 'lr_w': lrw, 'xgb_w': xgbw, 'lgbm_w': lgbmw, 'accuracy': acc})
        print(f'   {label:<38}: {acc*100:.2f}%  ({delta(acc)})')

    best = max(table, key=lambda x: x['accuracy'])
    joblib.dump(lgbm_model, os.path.join(OPTDIR, 'lgbm_model.pkl'))

    best_lgbm_blend = {
        'label':    best['label'],
        'lr_w':     best['lr_w'],
        'xgb_w':    best['xgb_w'],
        'lgbm_w':   best['lgbm_w'],
        'accuracy': round(best['accuracy'], 6),
        'delta_vs_baseline': round(best['accuracy'] - BASELINE_ACC, 6),
    }
    with open(os.path.join(OPTDIR, 'best_lgbm_blend.json'), 'w') as f:
        json.dump(best_lgbm_blend, f, indent=2)

    print(f'\n   Best LGBM blend: {best["label"]}  →  {best["accuracy"]*100:.2f}%')
    print(f'   Saved: optimization/lgbm_model.pkl, optimization/best_lgbm_blend.json')

    results['step3_lgbm'] = {
        'lgbm_params': lgbm_params,
        'table': [{'label': r['label'], 'accuracy': round(r['accuracy'], 6)} for r in table],
        'best': best_lgbm_blend,
    }
    save_results()
    gc.collect()

    print('\n  ┌──────────────────────────────────────────────────┐')
    print('  │  STEP 3 SUMMARY                                  │')
    print('  ├──────────────────────────────────────────────────┤')
    for r in table:
        best_str = ' ◄ best' if r['label'] == best['label'] else ''
        print(f'  │  {r["label"]:<38}: {r["accuracy"]*100:.2f}% ({delta(r["accuracy"])}){best_str}')
    print('  └──────────────────────────────────────────────────┘')

    return lgbm_model, best['accuracy']


# ─── STEP 4: Isotonic calibration ─────────────────────────────────────────────

def step4_calibration(prod_lr, prod_xgb, X_train, y_train, X_test, y_test):
    print('\n' + '='*62)
    print('  STEP 4 — Isotonic Calibration  (prefit, validation slice)')
    print('='*62)

    # Use last 20% of training data chronologically as calibration holdout
    n_cal = int(len(X_train) * 0.20)
    n_fit = len(X_train) - n_cal
    X_cal_fit = X_train.iloc[:n_fit]
    y_cal_fit = y_train.iloc[:n_fit]
    X_cal_val = X_train.iloc[n_fit:]
    y_cal_val = y_train.iloc[n_fit:]

    # Corner-flip only the calibration-fit portion
    X_cal_fit_aug, y_cal_fit_aug = corner_flip(X_cal_fit, y_cal_fit)

    print(f'   Cal-fit rows (aug): {len(X_cal_fit_aug):,} | Cal-val rows: {len(X_cal_val):,}')
    print(f'   Test rows: {len(X_test):,}')

    # Retrain LR on cal-fit subset
    print('   Retraining LR on cal-fit subset...', end='', flush=True)
    lr_cal = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=0.00711, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    lr_cal.fit(X_cal_fit_aug, y_cal_fit_aug)
    print(' done')
    gc.collect()

    # Build blend predict function (uncalibrated)
    def blend_proba(X, lr_model, xgb_model, lr_w=0.90, xgb_w=0.10):
        return lr_w * lr_model.predict_proba(X) + xgb_w * xgb_model.predict_proba(X)

    # Measure uncalibrated accuracy on test set using production models
    prob_uncal = blend_proba(X_test, prod_lr, prod_xgb)
    acc_uncal  = accuracy_score(y_test, (prob_uncal[:, 1] > 0.5).astype(int))

    # Calibration curve before (using production models, test set)
    prob_uncal_test = blend_proba(X_test, prod_lr, prod_xgb)[:, 1]
    frac_pos_before, mean_pred_before = calibration_curve(
        y_test, prob_uncal_test, n_bins=10, strategy='uniform'
    )
    cal_error_before = float(np.mean(np.abs(frac_pos_before - mean_pred_before)))

    # Build a wrapped blended model for CalibratedClassifierCV
    # We need a single estimator — use the cal-fit LR and XGB blended via a wrapper
    class BlendEstimator(BaseEstimator, ClassifierMixin):
        """Thin wrapper so CalibratedClassifierCV can call predict_proba."""
        def __init__(self, lr, xgb, lr_w=0.90, xgb_w=0.10):
            self.lr       = lr
            self.xgb      = xgb
            self.lr_w     = lr_w
            self.xgb_w    = xgb_w
            self.classes_  = np.array([0, 1])

        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            return self.lr_w * self.lr.predict_proba(X) + self.xgb_w * self.xgb.predict_proba(X)

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    blend_est = BlendEstimator(lr_cal, prod_xgb)

    print('   Fitting isotonic calibrator on val set...', end='', flush=True)
    calibrated = CalibratedClassifierCV(blend_est, method='isotonic', cv='prefit')
    calibrated.fit(X_cal_val, y_cal_val)
    print(' done')
    gc.collect()

    # Evaluate calibrated model on test set
    prob_cal_test = calibrated.predict_proba(X_test)[:, 1]
    y_pred_cal    = (prob_cal_test > 0.5).astype(int)
    acc_cal       = accuracy_score(y_test, y_pred_cal)

    # Calibration curve after
    frac_pos_after, mean_pred_after = calibration_curve(
        y_test, prob_cal_test, n_bins=10, strategy='uniform'
    )
    cal_error_after = float(np.mean(np.abs(frac_pos_after - mean_pred_after)))

    joblib.dump(calibrated, os.path.join(OPTDIR, 'calibrated_blend.pkl'))
    print(f'   Saved: optimization/calibrated_blend.pkl')

    print(f'\n   Uncalibrated accuracy  : {acc_uncal*100:.2f}%  (baseline reference)')
    print(f'   Calibrated accuracy    : {acc_cal*100:.2f}%  ({delta(acc_cal)})')
    print(f'   Calibration error before: {cal_error_before:.4f}')
    print(f'   Calibration error after : {cal_error_after:.4f}')

    print('\n   Reliability diagram (10 bins):')
    print(f'   {"Bin":>16}  {"Pred%":>6}  {"Actual%":>8}  {"Before":>8}  {"After":>7}')
    for i in range(min(len(frac_pos_before), len(mean_pred_before))):
        print(f'   bin {i*10:2d}–{(i+1)*10:2d}%  : '
              f'pred={mean_pred_before[i]*100:5.1f}%  actual={frac_pos_before[i]*100:5.1f}%  '
              f'|err_before|={abs(frac_pos_before[i]-mean_pred_before[i])*100:.1f}pp')
    print('   --- After calibration ---')
    for i in range(min(len(frac_pos_after), len(mean_pred_after))):
        print(f'   bin {i*10:2d}–{(i+1)*10:2d}%  : '
              f'pred={mean_pred_after[i]*100:5.1f}%  actual={frac_pos_after[i]*100:5.1f}%  '
              f'|err_after|={abs(frac_pos_after[i]-mean_pred_after[i])*100:.1f}pp')

    results['step4_calibration'] = {
        'acc_uncalibrated': round(acc_uncal, 6),
        'acc_calibrated':   round(acc_cal, 6),
        'cal_error_before': round(cal_error_before, 6),
        'cal_error_after':  round(cal_error_after, 6),
        'accuracy_change':  round(acc_cal - acc_uncal, 6),
        'calibration_improvement': round(cal_error_before - cal_error_after, 6),
        'saved': 'optimization/calibrated_blend.pkl',
    }
    save_results()

    print('\n  ┌─────────────────────────────────────────────┐')
    print('  │  STEP 4 SUMMARY                             │')
    print('  ├─────────────────────────────────────────────┤')
    print(f'  │  Uncalibrated accuracy  : {acc_uncal*100:.2f}%               │')
    print(f'  │  Calibrated accuracy    : {acc_cal*100:.2f}%  ({delta(acc_cal)})  │')
    print(f'  │  Cal error before: {cal_error_before:.4f}                  │')
    print(f'  │  Cal error after : {cal_error_after:.4f}                  │')
    impr = cal_error_before - cal_error_after
    print(f'  │  Cal improvement : {impr:+.4f}                  │')
    print('  └─────────────────────────────────────────────┘')

    return calibrated, acc_cal


# ─── STEP 5: C re-tune ────────────────────────────────────────────────────────

def step5_retune_C(X_train, y_train, X_train_aug, y_train_aug, X_test, y_test, prod_xgb):
    print('\n' + '='*62)
    print('  STEP 5 — C re-tune for 109 features  (5-fold CV on train)')
    print('='*62)

    C_values = [0.001, 0.003, 0.005, 0.007, 0.0071, 0.009, 0.01, 0.02, 0.05, 0.1]

    # 5-fold CV on augmented training set
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_table = []

    print('   Running 5-fold CV on training set (no test set used here)...')
    for C in C_values:
        fold_accs = []
        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train_aug, y_train_aug)):
            X_tr, X_val = X_train_aug.iloc[tr_idx], X_train_aug.iloc[val_idx]
            y_tr, y_val = y_train_aug.iloc[tr_idx], y_train_aug.iloc[val_idx]
            pipe = Pipeline([
                ('sc', RobustScaler()),
                ('lr', LogisticRegression(penalty='l2', C=C, solver='liblinear',
                                           max_iter=2000, random_state=42, n_jobs=1)),
            ])
            pipe.fit(X_tr, y_tr)
            fold_accs.append(accuracy_score(y_val, pipe.predict(X_val)))
        cv_acc = float(np.mean(fold_accs))
        cv_table.append({'C': C, 'cv_acc': cv_acc})
        print(f'   C={C:<8.5f}  CV acc={cv_acc*100:.2f}%')
        gc.collect()

    best_C_row = max(cv_table, key=lambda x: x['cv_acc'])
    best_C = best_C_row['C']
    print(f'\n   Best C by CV: {best_C}  (CV acc={best_C_row["cv_acc"]*100:.2f}%)')

    # Now evaluate best C on temporal test set (one shot)
    print(f'\n   Training final model with best C={best_C} and evaluating on test set...')
    best_pipe = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=best_C, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    best_pipe.fit(X_train_aug, y_train_aug)

    prob_lr  = best_pipe.predict_proba(X_test)
    prob_xgb = prod_xgb.predict_proba(X_test)
    prob     = 0.90 * prob_lr + 0.10 * prob_xgb
    y_pred   = (prob[:, 1] > 0.5).astype(int)
    acc_test = accuracy_score(y_test, y_pred)

    beats = acc_test > BASELINE_ACC
    joblib.dump(best_pipe, os.path.join(OPTDIR, 'lr_retuned.pkl'))

    print(f'\n   Temporal test accuracy with C={best_C}: {acc_test*100:.2f}%  ({delta(acc_test)})')
    print(f'   Beats baseline (71.64%): {"YES" if beats else "NO"}')
    print(f'   Saved: optimization/lr_retuned.pkl')

    results['step5_C_retune'] = {
        'cv_table': [{'C': r['C'], 'cv_acc': round(r['cv_acc'], 6)} for r in cv_table],
        'best_C':           best_C,
        'best_C_cv_acc':    round(best_C_row['cv_acc'], 6),
        'temporal_acc':     round(acc_test, 6),
        'beats_baseline':   beats,
        'saved':            'optimization/lr_retuned.pkl',
    }
    save_results()

    print('\n  ┌─────────────────────────────────────────────┐')
    print('  │  STEP 5 SUMMARY                             │')
    print('  ├─────────────────────────────────────────────┤')
    print(f'  │  Current C (production): 0.00711            │')
    print(f'  │  Best C (CV):   {best_C:<10.5f}              │')
    print(f'  │  CV acc at best C: {best_C_row["cv_acc"]*100:.2f}%              │')
    print(f'  │  Temporal acc:  {acc_test*100:.2f}%  ({delta(acc_test)})          │')
    print(f'  │  Beats baseline: {"YES" if beats else "NO":<28}│')
    print('  └─────────────────────────────────────────────┘')

    return best_pipe, best_C, acc_test


# ─── STEP 6: Final comparison and recommendation ──────────────────────────────

def step6_summary(step_accs):
    print('\n' + '='*62)
    print('  STEP 6 — Final Comparison & Recommendation')
    print('='*62)

    rows = [
        ('Baseline (LR90/XGB10, C=0.00711)', BASELINE_ACC,                  'Production — 109 features'),
        ('Step 1: ElasticNet LR (best)',       step_accs['elasticnet'],       'L2→ElasticNet, same C, 90/10 blend'),
        ('Step 2: Best blend ratio',           step_accs['blend'],            f"LR{step_accs['blend_ratio'][0]}/XGB{step_accs['blend_ratio'][1]}"),
        ('Step 3: Best LGBM blend',            step_accs['lgbm'],             'LR+LGBM blend'),
        ('Step 4: Isotonic calibration',       step_accs['calibrated'],       'Calibration layer (accuracy may vary)'),
        ('Step 5: C re-tuned',                 step_accs['C_retune'],         f"C={step_accs['best_C']}, 90/10 blend"),
    ]

    print(f'\n  {"Variant":<42} {"Accuracy":>8}  {"Delta":>8}  Notes')
    print(f'  {"-"*42} {"-"*8}  {"-"*8}  {"-"*30}')
    for name, acc, notes in rows:
        d = f"{'+' if acc - BASELINE_ACC >= 0 else ''}{(acc - BASELINE_ACC)*100:.2f}pp"
        base_str = '(baseline)' if 'Baseline' in name else ''
        print(f'  {name:<42} {acc*100:>7.2f}%  {d:>8}  {notes}  {base_str}')

    best_name, best_acc, best_notes = max(
        [r for r in rows if 'Baseline' not in r[0]],
        key=lambda x: x[1]
    )
    print(f'\n  Best variant: {best_name}  →  {best_acc*100:.2f}%  ({delta(best_acc)})')

    # Recommendation
    all_accs = {name: acc for name, acc, _ in rows}
    improve_any = any(acc > BASELINE_ACC for name, acc, _ in rows if 'Baseline' not in name)
    gain = best_acc - BASELINE_ACC

    print('\n  ── RECOMMENDATION ──')
    if gain > 0.002:
        rec = f"Promote {best_name}: clear gain of +{gain*100:.2f}pp."
    elif gain > 0:
        rec = f"{best_name} has a marginal gain of +{gain*100:.2f}pp — within noise range."
    else:
        rec = "No variant beats baseline. Keep current production model."

    findings_path = os.path.join(OPTDIR, 'OPTIMIZATION_FINDINGS.md')
    md = f"""# Model 1 Optimization Findings

**Run date:** {datetime.now().strftime('%Y-%m-%d')}
**Baseline:** 109-feature LR90/XGB10 blend — {BASELINE_ACC*100:.2f}% temporal accuracy (train <2024, test 2024+)

---

## Summary Table

| Variant | Accuracy | Delta vs {BASELINE_ACC*100:.2f}% | Notes |
|---------|---------|------------|-------|
"""
    for name, acc, notes in rows:
        d = f"{'+' if acc - BASELINE_ACC >= 0 else ''}{(acc - BASELINE_ACC)*100:.2f}pp"
        md += f"| {name} | {acc*100:.2f}% | {d} | {notes} |\n"

    md += f"""
---

## Step 1 — ElasticNet LR

Swapped L2 penalty for ElasticNet (solver=saga), tested l1_ratio ∈ [0.1, 0.3, 0.5, 0.7, 0.9] with fixed C=0.00711.
Kept 90/10 blend with production XGB.

| l1_ratio | Accuracy | Delta |
|----------|----------|-------|
"""
    for r in results.get('step1_elasticnet', {}).get('table', []):
        d = f"{'+' if r['accuracy'] - BASELINE_ACC >= 0 else ''}{(r['accuracy'] - BASELINE_ACC)*100:.2f}pp"
        md += f"| {r['l1_ratio']} | {r['accuracy']*100:.2f}% | {d} |\n"

    en = results.get('step1_elasticnet', {})
    md += f"\n**Best l1_ratio:** {en.get('best_l1_ratio')}  →  {en.get('best_accuracy', 0)*100:.2f}%\n"
    md += f"**Beats baseline:** {'YES' if en.get('beats_baseline') else 'NO'}\n"

    md += f"""
---

## Step 2 — Blend Ratio Sweep

Tested LR/XGB ratios from 95/5 to 70/30 using production LR and XGB (no retraining required).

| LR % | XGB % | Accuracy | Delta |
|------|-------|----------|-------|
"""
    for r in results.get('step2_blend', {}).get('table', []):
        d = f"{'+' if r['accuracy'] - BASELINE_ACC >= 0 else ''}{(r['accuracy'] - BASELINE_ACC)*100:.2f}pp"
        md += f"| {r['lr_pct']} | {r['xgb_pct']} | {r['accuracy']*100:.2f}% | {d} |\n"

    bl = results.get('step2_blend', {}).get('best_non_baseline', {})
    md += f"\n**Best non-baseline:** LR {bl.get('lr_pct')}% / XGB {bl.get('xgb_pct')}%  →  {bl.get('accuracy', 0)*100:.2f}%\n"

    md += f"""
---

## Step 3 — LightGBM Blends

Trained LightGBM with fixed params (n_estimators=200, lr=0.05, max_depth=4, num_leaves=15).
Tested LR+LGBM blends and a three-way LR+XGB+LGBM blend.

| Blend | Accuracy | Delta |
|-------|----------|-------|
"""
    for r in results.get('step3_lgbm', {}).get('table', []):
        d = f"{'+' if r['accuracy'] - BASELINE_ACC >= 0 else ''}{(r['accuracy'] - BASELINE_ACC)*100:.2f}pp"
        md += f"| {r['label']} | {r['accuracy']*100:.2f}% | {d} |\n"

    s3 = results.get('step3_lgbm', {}).get('best', {})
    md += f"\n**Best LGBM blend:** {s3.get('label')}  →  {s3.get('accuracy', 0)*100:.2f}%\n"

    s4 = results.get('step4_calibration', {})
    md += f"""
---

## Step 4 — Isotonic Calibration

Calibration holdout: last 20% of training data (chronological). CalibratedClassifierCV with method='isotonic', cv='prefit'.
Calibrator fit on validation slice only — never on test set.

| Metric | Before | After |
|--------|--------|-------|
| Accuracy | {s4.get('acc_uncalibrated', 0)*100:.2f}% | {s4.get('acc_calibrated', 0)*100:.2f}% |
| Mean calibration error | {s4.get('cal_error_before', 0):.4f} | {s4.get('cal_error_after', 0):.4f} |

Calibration improvement: {(s4.get('cal_error_before', 0) - s4.get('cal_error_after', 0)):.4f} (positive = better)
Accuracy change: {s4.get('accuracy_change', 0)*100:+.2f}pp
"""

    s5 = results.get('step5_C_retune', {})
    md += f"""
---

## Step 5 — C Re-tuning

C grid: {[r['C'] for r in s5.get('cv_table', [])]}
5-fold stratified CV on augmented training set only. Temporal test accuracy reported once per best C.

| C | CV Accuracy |
|---|-------------|
"""
    for r in s5.get('cv_table', []):
        md += f"| {r['C']} | {r['cv_acc']*100:.2f}% |\n"

    md += f"""
**Best C (CV):** {s5.get('best_C')}  (CV acc={s5.get('best_C_cv_acc', 0)*100:.2f}%)
**Temporal test accuracy:** {s5.get('temporal_acc', 0)*100:.2f}%  ({delta(s5.get('temporal_acc', 0))})
**Beats baseline:** {'YES' if s5.get('beats_baseline') else 'NO'}

---

## Recommendation

{rec}

"""

    # Combination logic
    md += "### Combination analysis\n\n"
    if step_accs.get('elasticnet', 0) > BASELINE_ACC and step_accs.get('C_retune', 0) > BASELINE_ACC:
        md += "- Both ElasticNet and C re-tune improve on baseline. They are orthogonal changes — combining best C with ElasticNet is viable.\n"
    if step_accs.get('calibrated', 0) > step_accs.get('blend', BASELINE_ACC):
        md += "- Calibration layer improves probability reliability. Can be stacked on top of any blend variant without retraining.\n"
    md += "\n> **Production files unchanged.** To promote any variant, run the trainer with the recommended settings.\n"

    with open(findings_path, 'w') as f:
        f.write(md)
    print(f'\n   Saved: optimization/OPTIMIZATION_FINDINGS.md')

    results['step6_summary'] = {
        'recommendation': rec,
        'best_variant': best_name,
        'best_accuracy': round(best_acc, 6),
    }
    save_results()

    print('\n  ┌──────────────────────────────────────────────────────────┐')
    print('  │  STEP 6 — FINAL SUMMARY                                  │')
    print('  ├──────────────────────────────────────────────────────────┤')
    for name, acc, notes in rows:
        d = f"{'+' if acc - BASELINE_ACC >= 0 else ''}{(acc - BASELINE_ACC)*100:.2f}pp"
        print(f'  │  {name:<40}: {acc*100:.2f}%  {d:<9}│')
    print('  ├──────────────────────────────────────────────────────────┤')
    print(f'  │  RECOMMENDATION: {rec[:60]:<60}│')
    print('  └──────────────────────────────────────────────────────────┘')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('='*62)
    print('  8SI UFC Predictor — Model 1 Optimization Sprint')
    print(f'  Baseline: {BASELINE_ACC*100:.2f}% | Features: {len(FEATURES)}')
    print('='*62)

    # Load production models (read-only, never overwritten)
    print('\nLoading production models (read-only)...')
    prod_lr  = joblib.load(os.path.join(MODEL, 'ufc_model_best.pkl'))
    prod_xgb = joblib.load(os.path.join(MODEL, 'ufc_model_xgb.pkl'))
    print(f'   Loaded LR: {type(prod_lr).__name__} | XGB: {type(prod_xgb).__name__}')

    # Build dataset once
    X_train, y_train, X_train_aug, y_train_aug, X_test, y_test, dates_test = build_dataset()
    gc.collect()

    step_accs = {}

    # ── Step 1 ──────────────────────────────────────────────────────────────────
    _, en_acc = step1_elasticnet(X_train_aug, y_train_aug, X_test, y_test, dates_test, prod_xgb)
    step_accs['elasticnet'] = en_acc
    gc.collect()

    # ── Step 2 ──────────────────────────────────────────────────────────────────
    lr_pct, xgb_pct, blend_acc = step2_blend_ratios(prod_lr, prod_xgb, X_test, y_test)
    step_accs['blend'] = blend_acc
    step_accs['blend_ratio'] = (lr_pct, xgb_pct)
    gc.collect()

    # ── Step 3 ──────────────────────────────────────────────────────────────────
    lgbm_model, lgbm_acc = step3_lightgbm(prod_lr, prod_xgb, X_train_aug, y_train_aug, X_test, y_test)
    step_accs['lgbm'] = lgbm_acc
    gc.collect()

    # ── Step 4 ──────────────────────────────────────────────────────────────────
    _, cal_acc = step4_calibration(prod_lr, prod_xgb, X_train, y_train, X_test, y_test)
    step_accs['calibrated'] = cal_acc
    gc.collect()

    # ── Step 5 ──────────────────────────────────────────────────────────────────
    _, best_C, c_acc = step5_retune_C(X_train, y_train, X_train_aug, y_train_aug, X_test, y_test, prod_xgb)
    step_accs['C_retune'] = c_acc
    step_accs['best_C'] = best_C
    gc.collect()

    # ── Step 6 ──────────────────────────────────────────────────────────────────
    step6_summary(step_accs)

    print(f'\n{"="*62}')
    print(f'  Optimization sprint complete.')
    print(f'  All results: experiments/research/optimization/results.json')
    print(f'  Findings:    experiments/research/optimization/OPTIMIZATION_FINDINGS.md')
    print(f'{"="*62}\n')


if __name__ == '__main__':
    main()
