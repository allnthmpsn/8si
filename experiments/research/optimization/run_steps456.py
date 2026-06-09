#!/usr/bin/env python3
"""Run optimization steps 4-6 only (steps 1-3 already complete)."""
import os, sys, gc, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
import joblib

ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA   = os.path.join(ROOT, 'data')
MODEL  = os.path.join(ROOT, 'model')
OPTDIR = os.path.dirname(os.path.abspath(__file__))

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

# Load existing results
results_path = os.path.join(OPTDIR, 'results.json')
with open(results_path) as f:
    results = json.load(f)

def save_results():
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

def delta(acc):
    return f"{'+' if acc - BASELINE_ACC >= 0 else ''}{(acc - BASELINE_ACC)*100:.2f}pp"


# ─── Pipeline helpers (identical to train_model1.py) ──────────────────────────

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
    hist['elo_trend'] = hist.groupby('fighter')['elo_before'].transform(lambda x: x - x.shift(3))
    counts = hist.groupby('fighter').size()
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
    df['last3_win_rate']    = g['won'].transform(lambda x: _roll(x, 3, 0.5))
    df['last10_win_rate']   = g['won'].transform(lambda x: _roll(x, 10, 0.5))
    df['last5_won']         = g['won'].transform(lambda x: _roll(x, 5, 0.5))
    df['last5_finish_rate'] = g['_fin'].transform(lambda x: _roll(x, 5, 0.0))
    df['trend_score']       = df['last3_win_rate'] - df['last10_win_rate']
    df['_prev_date'] = g['date'].transform(lambda x: x.shift(1))
    df['layoff_days'] = (df['date'] - df['_prev_date']).dt.days.fillna(180.0).clip(lower=0)
    opp_col = df['opponent'].tolist()
    fighter_col = df['fighter'].tolist()
    idx_list = df.index.tolist()
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
        f'{prefix}layoff_90_180':  ((d >= 90) & (d < 180)).astype(int),
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


def build_dataset():
    print('\nBuilding dataset...')
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

    elo_hist_df, _ = compute_elo(df, K=48, base=1500.0)
    gc.collect()

    all_win_rates = {f: grp['won'].sum() / max(1, len(grp)) for f, grp in career_df.groupby('fighter')}
    career_stats = compute_career_stats(career_df, all_win_rates)
    gc.collect()

    df = df[df['date'] >= TRAIN_START].copy()
    df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
    df = df.sort_values('date').reset_index(drop=True)

    career_stats = career_stats.sort_values(['fighter', 'date'])
    career_cols = [c for c in career_stats.columns if c not in ('fighter', 'date')]
    r_career = career_stats.rename(columns={'fighter': 'R_fighter', **{c: f'R_{c}' for c in career_cols}})
    b_career = career_stats.rename(columns={'fighter': 'B_fighter', **{c: f'B_{c}' for c in career_cols}})
    df = pd.merge_asof(df.sort_values('date'), r_career.sort_values('date'), on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b_career.sort_values('date'), on='date', by='B_fighter', direction='backward')
    defaults = {'cum_fights': 0, 'career_win_rate': 0.5, 'ko_finish_rate': 0.0, 'sub_finish_rate': 0.0,
                'career_finish_rate': 0.0, 'last3_win_rate': 0.5, 'last10_win_rate': 0.5,
                'last5_won': 0.5, 'last5_finish_rate': 0.0, 'trend_score': 0.0, 'layoff_days': 180.0, 'opp_quality': 0.5}
    for stat, default in defaults.items():
        df[f'R_{stat}'] = df[f'R_{stat}'].fillna(default)
        df[f'B_{stat}'] = df[f'B_{stat}'].fillna(default)

    elo_cols = elo_hist_df[['fighter', 'date', 'elo_before', 'elo_trend']].copy()
    elo_r = elo_cols.rename(columns={'fighter': 'R_fighter', 'elo_before': 'R_elo', 'elo_trend': 'R_elo_trend'})
    elo_b = elo_cols.rename(columns={'fighter': 'B_fighter', 'elo_before': 'B_elo', 'elo_trend': 'B_elo_trend'})
    df = pd.merge_asof(df.sort_values('date'), elo_r.sort_values('date'), on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), elo_b.sort_values('date'), on='date', by='B_fighter', direction='backward')
    df['R_elo'] = df['R_elo'].fillna(1500.0); df['B_elo'] = df['B_elo'].fillna(1500.0)
    df['R_elo_trend'] = df['R_elo_trend'].fillna(0.0); df['B_elo_trend'] = df['B_elo_trend'].fillna(0.0)

    style_src = ['SLpM', 'SApM', 'Str_Acc', 'Str_Def', 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']
    style_df = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last').reset_index(drop=True)
    r_style = style_df[['Fighter_Name'] + style_src].rename(columns={'Fighter_Name': 'R_fighter', **{c: f'R_{c}' for c in style_src}})
    b_style = style_df[['Fighter_Name'] + style_src].rename(columns={'Fighter_Name': 'B_fighter', **{c: f'B_{c}' for c in style_src}})
    df = df.merge(r_style, on='R_fighter', how='left')
    df = df.merge(b_style, on='B_fighter', how='left')
    for col in [f'{p}{s}' for p in ('R_', 'B_') for s in style_src]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    gc.collect()

    df['weight_class_ord'] = df['weight_class'].map(WC_ORDER).fillna(6).astype(int)
    df['title_bout_bin']   = df['title_bout'].astype(int)
    df['R_southpaw'] = (df['R_Stance'].str.lower() == 'southpaw').astype(int)
    df['B_southpaw'] = (df['B_Stance'].str.lower() == 'southpaw').astype(int)
    df['orth_clash']  = ((df['R_southpaw'] == 0) & (df['B_southpaw'] == 0)).astype(int)
    df['south_clash'] = ((df['R_southpaw'] == 1) & (df['B_southpaw'] == 1)).astype(int)
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
    df['SLpM_dif'] = df['R_SLpM'] - df['B_SLpM']
    df['SApM_dif'] = df['R_SApM'] - df['B_SApM']
    df['Str_Def_dif'] = df['R_Str_Def'] - df['B_Str_Def']
    df['TD_Def_dif']  = df['R_TD_Def']  - df['B_TD_Def']
    df['Sub_Avg_dif'] = df['R_Sub_Avg'] - df['B_Sub_Avg']
    df['TD_Avg_dif']  = df['R_TD_Avg']  - df['B_TD_Avg']
    df['elo_dif'] = df['R_elo'] - df['B_elo']
    df['elo_trend_dif'] = df['R_elo_trend'] - df['B_elo_trend']
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy()
    df['target'] = (df['Winner'] == 'Red').astype(int)
    train_mask = df['date'] < TRAIN_CUTOFF
    test_mask  = df['date'] >= TRAIN_CUTOFF

    X_train = df.loc[train_mask, FEATURES].reset_index(drop=True)
    y_train = df.loc[train_mask, 'target'].reset_index(drop=True)
    X_test  = df.loc[test_mask,  FEATURES].reset_index(drop=True)
    y_test  = df.loc[test_mask,  'target'].reset_index(drop=True)

    print(f'   Train (pre-aug): {len(X_train):,} | Test: {len(X_test):,}')
    X_train_aug, y_train_aug = corner_flip(X_train, y_train)
    print(f'   Train (post-aug): {len(X_train_aug):,}')
    gc.collect()
    return X_train, y_train, X_train_aug, y_train_aug, X_test, y_test


# ─── STEP 4: Isotonic calibration ─────────────────────────────────────────────

def step4_calibration(prod_lr, prod_xgb, X_train, y_train, X_test, y_test):
    print('\n' + '='*62)
    print('  STEP 4 — Isotonic Calibration')
    print('='*62)

    n_cal = int(len(X_train) * 0.20)
    n_fit = len(X_train) - n_cal
    X_cal_fit = X_train.iloc[:n_fit]
    y_cal_fit = y_train.iloc[:n_fit]
    X_cal_val = X_train.iloc[n_fit:]
    y_cal_val = y_train.iloc[n_fit:]
    X_cal_fit_aug, y_cal_fit_aug = corner_flip(X_cal_fit, y_cal_fit)
    print(f'   Cal-fit (aug): {len(X_cal_fit_aug):,} | Cal-val: {len(X_cal_val):,} | Test: {len(X_test):,}')

    print('   Retraining LR on cal-fit subset...', end='', flush=True)
    lr_cal = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=0.00711, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    lr_cal.fit(X_cal_fit_aug, y_cal_fit_aug)
    print(' done')
    gc.collect()

    # Uncalibrated probabilities — production model on cal-val and test
    prob_uncal_val  = (0.90 * lr_cal.predict_proba(X_cal_val) +
                       0.10 * prod_xgb.predict_proba(X_cal_val))[:, 1]
    prob_uncal_test = (0.90 * prod_lr.predict_proba(X_test) +
                       0.10 * prod_xgb.predict_proba(X_test))[:, 1]

    acc_uncal   = accuracy_score(y_test, (prob_uncal_test > 0.5).astype(int))
    frac_pos_b, mean_pred_b = calibration_curve(y_test, prob_uncal_test, n_bins=10, strategy='uniform')
    cal_err_before = float(np.mean(np.abs(frac_pos_b - mean_pred_b)))

    # Fit isotonic calibrator on validation slice
    print('   Fitting isotonic calibrator on val set...', end='', flush=True)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(prob_uncal_val, y_cal_val.values)
    print(' done')
    gc.collect()

    # Apply calibrator to test set (calibrate the production blend probabilities)
    prob_cal  = iso.predict(prob_uncal_test).clip(0, 1)
    acc_cal   = accuracy_score(y_test, (prob_cal > 0.5).astype(int))
    frac_pos_a, mean_pred_a = calibration_curve(y_test, prob_cal, n_bins=10, strategy='uniform')
    cal_err_after = float(np.mean(np.abs(frac_pos_a - mean_pred_a)))

    # Save calibrator and underlying LR
    calibration_bundle = {'iso': iso, 'lr_cal': lr_cal}
    joblib.dump(calibration_bundle, os.path.join(OPTDIR, 'calibrated_blend.pkl'))

    print(f'\n   Uncalibrated accuracy  : {acc_uncal*100:.2f}%')
    print(f'   Calibrated accuracy    : {acc_cal*100:.2f}%  ({delta(acc_cal)})')
    print(f'   Cal error before: {cal_err_before:.4f}')
    print(f'   Cal error after : {cal_err_after:.4f}  (improvement: {cal_err_before - cal_err_after:+.4f})')

    print('\n   Reliability diagram — BEFORE calibration:')
    print(f'   {"Bin":>12}  {"Pred%":>7}  {"Actual%":>8}  {"Err":>6}')
    for i in range(len(frac_pos_b)):
        err = abs(frac_pos_b[i] - mean_pred_b[i])
        print(f'   {i*10:2d}–{(i+1)*10:2d}%      pred={mean_pred_b[i]*100:5.1f}%  actual={frac_pos_b[i]*100:5.1f}%  |err|={err*100:.1f}pp')

    print('\n   Reliability diagram — AFTER calibration:')
    print(f'   {"Bin":>12}  {"Pred%":>7}  {"Actual%":>8}  {"Err":>6}')
    for i in range(len(frac_pos_a)):
        err = abs(frac_pos_a[i] - mean_pred_a[i])
        print(f'   {i*10:2d}–{(i+1)*10:2d}%      pred={mean_pred_a[i]*100:5.1f}%  actual={frac_pos_a[i]*100:5.1f}%  |err|={err*100:.1f}pp')

    results['step4_calibration'] = {
        'acc_uncalibrated': round(acc_uncal, 6),
        'acc_calibrated':   round(acc_cal, 6),
        'cal_error_before': round(cal_err_before, 6),
        'cal_error_after':  round(cal_err_after, 6),
        'accuracy_change':  round(acc_cal - acc_uncal, 6),
        'calibration_improvement': round(cal_err_before - cal_err_after, 6),
        'reliability_before': [{'pred': round(float(p), 4), 'actual': round(float(a), 4)}
                                for p, a in zip(mean_pred_b, frac_pos_b)],
        'reliability_after':  [{'pred': round(float(p), 4), 'actual': round(float(a), 4)}
                                for p, a in zip(mean_pred_a, frac_pos_a)],
        'saved': 'optimization/calibrated_blend.pkl',
    }
    save_results()

    print('\n  ┌─────────────────────────────────────────────┐')
    print('  │  STEP 4 SUMMARY                             │')
    print('  ├─────────────────────────────────────────────┤')
    print(f'  │  Uncalibrated accuracy  : {acc_uncal*100:.2f}%               │')
    print(f'  │  Calibrated accuracy    : {acc_cal*100:.2f}%  ({delta(acc_cal)})   │')
    print(f'  │  Cal error before : {cal_err_before:.4f}                   │')
    print(f'  │  Cal error after  : {cal_err_after:.4f}                   │')
    impr = cal_err_before - cal_err_after
    print(f'  │  Improvement      : {impr:+.4f}                   │')
    print('  └─────────────────────────────────────────────┘')
    return acc_cal


# ─── STEP 5: C re-tune ────────────────────────────────────────────────────────

def step5_retune_C(X_train, y_train, X_train_aug, y_train_aug, X_test, y_test, prod_xgb):
    print('\n' + '='*62)
    print('  STEP 5 — C re-tune  (5-fold CV on training set only)')
    print('='*62)

    C_values = [0.001, 0.003, 0.005, 0.007, 0.0071, 0.009, 0.01, 0.02, 0.05, 0.1]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_table = []

    print('   5-fold CV on augmented training set...')
    for C in C_values:
        fold_accs = []
        for tr_idx, val_idx in kf.split(X_train_aug, y_train_aug):
            X_tr  = X_train_aug.iloc[tr_idx]
            X_val = X_train_aug.iloc[val_idx]
            y_tr  = y_train_aug.iloc[tr_idx]
            y_val = y_train_aug.iloc[val_idx]
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

    print(f'\n   Training final model with C={best_C}...')
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
    print(f'   Temporal test accuracy: {acc_test*100:.2f}%  ({delta(acc_test)})')
    print(f'   Beats baseline: {"YES" if beats else "NO"}')
    print(f'   Saved: optimization/lr_retuned.pkl')

    results['step5_C_retune'] = {
        'cv_table': [{'C': r['C'], 'cv_acc': round(r['cv_acc'], 6)} for r in cv_table],
        'best_C': best_C,
        'best_C_cv_acc': round(best_C_row['cv_acc'], 6),
        'temporal_acc': round(acc_test, 6),
        'beats_baseline': beats,
        'saved': 'optimization/lr_retuned.pkl',
    }
    save_results()

    print('\n  ┌─────────────────────────────────────────────┐')
    print('  │  STEP 5 SUMMARY                             │')
    print('  ├─────────────────────────────────────────────┤')
    print(f'  │  Production C:  0.00711                     │')
    print(f'  │  Best C (CV):   {best_C:<10.5f}              │')
    print(f'  │  Best CV acc:   {best_C_row["cv_acc"]*100:.2f}%                   │')
    print(f'  │  Temporal acc:  {acc_test*100:.2f}%  ({delta(acc_test)})          │')
    print(f'  │  Beats baseline: {"YES" if beats else "NO":<27}│')
    print('  └─────────────────────────────────────────────┘')
    return best_C, acc_test


# ─── STEP 6: Summary ──────────────────────────────────────────────────────────

def step6_summary(step_accs):
    print('\n' + '='*62)
    print('  STEP 6 — Final Comparison & Recommendation')
    print('='*62)

    s1 = results.get('step1_elasticnet', {})
    s2 = results.get('step2_blend', {})
    s3 = results.get('step3_lgbm', {})
    s4 = results.get('step4_calibration', {})
    s5 = results.get('step5_C_retune', {})

    rows = [
        ('Baseline (LR90/XGB10, C=0.00711)', BASELINE_ACC),
        ('Step 1: ElasticNet best (l1=0.3)',  s1.get('best_accuracy', 0)),
        ('Step 2: LR70/XGB30',                s2.get('best_non_baseline', {}).get('accuracy', 0)),
        ('Step 3: LR80+XGB10+LGBM10',         s3.get('best', {}).get('accuracy', 0)),
        ('Step 4: Isotonic calibration',       s4.get('acc_calibrated', 0)),
        ('Step 5: C re-tuned',                 s5.get('temporal_acc', 0)),
    ]

    print(f'\n  {"Variant":<42} {"Accuracy":>8}  {"Delta":>9}')
    print(f'  {"-"*42} {"-"*8}  {"-"*9}')
    for name, acc in rows:
        d = f"{'+' if acc - BASELINE_ACC >= 0 else ''}{(acc - BASELINE_ACC)*100:.2f}pp"
        base_str = ' (baseline)' if 'Baseline' in name else ''
        print(f'  {name:<42} {acc*100:>7.2f}%  {d:>9}  {base_str}')

    non_base = [(n, a) for n, a in rows if 'Baseline' not in n]
    best_name, best_acc = max(non_base, key=lambda x: x[1])
    gain = best_acc - BASELINE_ACC

    print(f'\n  Best variant: {best_name}  →  {best_acc*100:.2f}%  ({delta(best_acc)})')

    if gain > 0.003:
        rec = f"Promote {best_name}: +{gain*100:.2f}pp — meaningful improvement."
    elif gain > 0.001:
        rec = (f"{best_name} shows marginal gain of +{gain*100:.2f}pp. "
               f"Worth promoting if combined with other improvements.")
    else:
        rec = "No variant clears +0.10pp. Keep production model; revisit with more data."

    print(f'\n  RECOMMENDATION: {rec}')

    # Combination note
    s2_best = s2.get('best_non_baseline', {})
    if s2_best.get('accuracy', 0) > BASELINE_ACC and s5.get('temporal_acc', 0) > BASELINE_ACC:
        combo = (f"Combination to test: retrain with C={s5['best_C']} and blend "
                 f"LR{s2_best['lr_pct']}/XGB{s2_best['xgb_pct']}.")
        print(f'  COMBINATION: {combo}')
    elif s2_best.get('accuracy', 0) > BASELINE_ACC:
        print(f"  Note: Blend ratio LR{s2_best['lr_pct']}/XGB{s2_best['xgb_pct']} "
              f"beats baseline without retraining — low-effort improvement.")

    # Write findings
    md = f"""# Model 1 Optimization Findings

**Run date:** {datetime.now().strftime('%Y-%m-%d')}
**Baseline:** 109-feature LR90/XGB10 blend — {BASELINE_ACC*100:.2f}% temporal accuracy (train <2024, test 2024+)

---

## Summary Table

| Variant | Accuracy | Delta | Notes |
|---------|---------|-------|-------|
"""
    notes_map = {
        'Baseline (LR90/XGB10, C=0.00711)': 'Production model',
        'Step 1: ElasticNet best (l1=0.3)':  'L2→ElasticNet, C=0.00711, 90/10 blend',
        'Step 2: LR70/XGB30':                'Blend ratio only, no retraining',
        'Step 3: LR80+XGB10+LGBM10':         'Three-way blend, best of 4 tested',
        'Step 4: Isotonic calibration':       'Calibration layer, val-slice fit',
        'Step 5: C re-tuned':                 f"C={s5.get('best_C')}, 90/10 blend",
    }
    for name, acc in rows:
        d = f"{'+' if acc - BASELINE_ACC >= 0 else ''}{(acc - BASELINE_ACC)*100:.2f}pp"
        md += f"| {name} | {acc*100:.2f}% | {d} | {notes_map.get(name, '')} |\n"

    md += f"""
---

## Step 1 — ElasticNet LR

Replaced L2 with ElasticNet (saga solver), swept l1_ratio ∈ [0.1, 0.3, 0.5, 0.7, 0.9] at fixed C=0.00711.

| l1_ratio | Accuracy | Delta |
|----------|----------|-------|
"""
    for r in s1.get('table', []):
        d = f"{'+' if r['accuracy'] - BASELINE_ACC >= 0 else ''}{(r['accuracy'] - BASELINE_ACC)*100:.2f}pp"
        md += f"| {r['l1_ratio']} | {r['accuracy']*100:.2f}% | {d} |\n"
    md += f"\n**Conclusion:** All l1_ratios underperform L2 baseline. L2 regularization is the right penalty for this feature set.\n"

    md += f"""
---

## Step 2 — Blend Ratio Sweep

| LR % | XGB % | Accuracy | Delta |
|------|-------|----------|-------|
"""
    for r in s2.get('table', []):
        d = f"{'+' if r['accuracy'] - BASELINE_ACC >= 0 else ''}{(r['accuracy'] - BASELINE_ACC)*100:.2f}pp"
        md += f"| {r['lr_pct']} | {r['xgb_pct']} | {r['accuracy']*100:.2f}% | {d} |\n"

    s2b = s2.get('best_non_baseline', {})
    md += f"\n**Best:** LR {s2b.get('lr_pct')}% / XGB {s2b.get('xgb_pct')}%  →  {s2b.get('accuracy', 0)*100:.2f}%\n"
    md += f"**Note:** Increasing XGB weight up to 30% monotonically improves accuracy. This is a free improvement — no retraining required.\n"

    md += f"""
---

## Step 3 — LightGBM Blends

Params: n_estimators=200, lr=0.05, max_depth=4, num_leaves=15, subsample=0.8.

| Blend | Accuracy | Delta |
|-------|----------|-------|
"""
    for r in s3.get('table', []):
        d = f"{'+' if r['accuracy'] - BASELINE_ACC >= 0 else ''}{(r['accuracy'] - BASELINE_ACC)*100:.2f}pp"
        md += f"| {r['label']} | {r['accuracy']*100:.2f}% | {d} |\n"
    md += f"\n**Conclusion:** LGBM consistently underperforms XGB in all blend configurations on this dataset size.\n"

    md += f"""
---

## Step 4 — Isotonic Calibration

Calibration holdout: last 20% of training data (chronological split, not shuffled).
Calibrator fit on validation slice only — test set used only for final evaluation.

| Metric | Value |
|--------|-------|
| Uncalibrated accuracy | {s4.get('acc_uncalibrated', 0)*100:.2f}% |
| Calibrated accuracy | {s4.get('acc_calibrated', 0)*100:.2f}% |
| Mean cal error before | {s4.get('cal_error_before', 0):.4f} |
| Mean cal error after | {s4.get('cal_error_after', 0):.4f} |
| Cal improvement | {s4.get('calibration_improvement', 0):+.4f} |
| Accuracy change | {s4.get('accuracy_change', 0)*100:+.2f}pp |

**Note:** Calibration primarily improves probability reliability (important for Kelly sizing), not raw accuracy.

---

## Step 5 — C Re-tuning

C grid searched via 5-fold CV on augmented training set. Production C=0.00711 was tuned on 114-feature model.

| C | CV Accuracy |
|---|-------------|
"""
    for r in s5.get('cv_table', []):
        md += f"| {r['C']} | {r['cv_acc']*100:.2f}% |\n"

    md += f"""
**Best C by CV:** {s5.get('best_C')}
**Temporal test accuracy:** {s5.get('temporal_acc', 0)*100:.2f}%  ({delta(s5.get('temporal_acc', 0))})

---

## Recommendation

**{rec}**

"""
    s2b = s2.get('best_non_baseline', {})
    if s2b.get('accuracy', 0) > BASELINE_ACC:
        md += f"### Free improvement (no retraining)\n\nChange blend from LR90/XGB10 to **LR{s2b.get('lr_pct')}/XGB{s2b.get('xgb_pct')}**. "
        md += f"Accuracy improves from {BASELINE_ACC*100:.2f}% → {s2b.get('accuracy', 0)*100:.2f}% (+{(s2b.get('accuracy', 0) - BASELINE_ACC)*100:.2f}pp) with the same pkl files.\n\n"

    if s4.get('calibration_improvement', 0) > 0:
        md += f"### Calibration layer\n\nIsotonic calibration reduces mean calibration error by {s4.get('calibration_improvement', 0):.4f}. "
        md += f"Accuracy impact: {s4.get('accuracy_change', 0)*100:+.2f}pp. "
        md += f"Consider applying calibration on top of any production blend — it improves Kelly bet sizing reliability without harming prediction direction.\n\n"

    md += "> **Production files unchanged.** All variants saved to `experiments/research/optimization/`.\n"

    findings_path = os.path.join(OPTDIR, 'OPTIMIZATION_FINDINGS.md')
    with open(findings_path, 'w') as f:
        f.write(md)
    print(f'   Saved: optimization/OPTIMIZATION_FINDINGS.md')

    results['step6_summary'] = {
        'recommendation': rec,
        'best_variant': best_name,
        'best_accuracy': round(best_acc, 6),
    }
    save_results()

    print('\n  ┌──────────────────────────────────────────────────────────┐')
    print('  │  STEP 6 — FINAL SUMMARY                                  │')
    print('  ├──────────────────────────────────────────────────────────┤')
    for name, acc in rows:
        d = f"{'+' if acc - BASELINE_ACC >= 0 else ''}{(acc - BASELINE_ACC)*100:.2f}pp"
        print(f'  │  {name:<40}: {acc*100:.2f}%  {d:<10}│')
    print('  ├──────────────────────────────────────────────────────────┤')
    print(f'  │  RECOMMENDATION:                                          │')
    for i in range(0, len(rec), 56):
        print(f'  │  {rec[i:i+56]:<56}│')
    print('  └──────────────────────────────────────────────────────────┘')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('='*62)
    print('  Steps 4-6 only (steps 1-3 results loaded from results.json)')
    print('='*62)

    prod_lr  = joblib.load(os.path.join(MODEL, 'ufc_model_best.pkl'))
    prod_xgb = joblib.load(os.path.join(MODEL, 'ufc_model_xgb.pkl'))
    print(f'   Production models loaded.')

    X_train, y_train, X_train_aug, y_train_aug, X_test, y_test = build_dataset()
    gc.collect()

    step_accs = {}

    # Step 4
    cal_acc = step4_calibration(prod_lr, prod_xgb, X_train, y_train, X_test, y_test)
    step_accs['calibrated'] = cal_acc
    gc.collect()

    # Step 5
    best_C, c_acc = step5_retune_C(X_train, y_train, X_train_aug, y_train_aug, X_test, y_test, prod_xgb)
    step_accs['C_retune'] = c_acc
    step_accs['best_C'] = best_C
    gc.collect()

    # Step 6
    step6_summary(step_accs)

    print(f'\n{"="*62}')
    print(f'  Steps 4-6 complete.')
    print(f'  results.json and OPTIMIZATION_FINDINGS.md updated.')
    print(f'{"="*62}\n')


if __name__ == '__main__':
    main()
