#!/usr/bin/env python3
"""
8SI UFC Predictor — Model 1 Production Trainer
Variant V2 (May 2026 sprint):
  Blend:   70% LogisticRegression + 30% XGBoost
  Features: 129 (109 base + 12 QA stats + 8 interaction features)
  Scope:   Men's weight classes only (women's excluded)
  Window:  2015-01-01 to 2023-12-31 (expanded from 2018)
  Recency: Exponential decay, half-life=730 days
  XGB params: Optuna-tuned, 25 trials, 5-fold CV
  Accuracy: 72.81% temporal (2024+ men's holdout)

Previous (Variant A, May 2026):
  109 features, all fights 2018+, 72.08% accuracy (mixed dataset)
"""
import sys
import os
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

# ─── Paths (resolve relative to this file so script is runnable from anywhere) ─
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(ROOT, 'data')
MODEL = os.path.join(ROOT, 'model')

TRAIN_START  = '2015-01-01'   # expanded from 2018 — data quality check passed
TRAIN_CUTOFF = '2024-01-01'   # train on [START, CUTOFF), test on [CUTOFF, ∞)
LR_WEIGHT    = 0.70
XGB_WEIGHT   = 0.30
HL_DAYS      = 730            # recency half-life in days (2 years)

WOMENS_CLASSES = [
    "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
]

# ─── 129-feature list (must match backend/main.py and feature_columns_best.pkl) ─
# Base 109 (Variant A) + 12 QA stats + 8 interaction features
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

# QA stats: opponent-elo-weighted career metrics (higher signal than raw)
FEAT_QA = [
    "R_qa_win_rate", "R_qa_finish_rate", "R_qa_SLpM", "R_qa_SApM",
    "B_qa_win_rate", "B_qa_finish_rate", "B_qa_SLpM", "B_qa_SApM",
    "qa_win_rate_dif", "qa_finish_rate_dif", "qa_SLpM_dif", "qa_SApM_dif",
]

# Interaction features: age×layoff, finishing danger, chin proxy
FEAT_INT = [
    "R_age_x_layoff", "B_age_x_layoff", "age_x_layoff_dif",
    "R_finish_danger", "B_finish_danger", "finish_danger_mismatch",
    "R_got_finished_rate", "B_got_finished_rate",
]

FEAT_114 = FEAT_BASE + FEAT_QA + FEAT_INT   # variable kept for compatibility

# Optuna-tuned XGB params (25 trials, 5-fold CV, May 2026 sprint)
XGB_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6,
    'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'gamma': 0.3, 'reg_alpha': 0, 'reg_lambda': 2.0,
    'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1,
}

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}


# ─────────────────────────────────────────────────────────────────────────────
# Elo computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_elo(df_all, K=48, base=1500.0):
    """
    Compute Elo from ufc-master.csv (one row per fight, both corners).
    Returns:
        history_df  — fighter, opponent, date, elo_before, elo_after, result, elo_trend
        current_df  — fighter, current_elo, last_fight_date, total_fights
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Career stats (shift=1, no leakage)
# ─────────────────────────────────────────────────────────────────────────────
def compute_career_stats(career_df, all_win_rates):
    """
    Per-fighter career stats computed with shift(1) — every stat row
    represents what was known BEFORE that fight.

    Uses vectorized cumsum-shift for rates and a tight loop for
    rolling-window and opp_quality stats.
    """
    df = career_df.sort_values(['fighter', 'date']).copy().reset_index(drop=True)

    # Finish type flags
    df['_ko']  = ((df['won'] == 1) & df['method'].str.contains('KO|TKO', case=False, na=False)).astype(float)
    df['_sub'] = ((df['won'] == 1) & df['method'].str.contains('Sub|Submission', case=False, na=False)).astype(float)
    df['_fin'] = ((df['won'] == 1) & df['method'].str.contains('KO|TKO|Sub', case=False, na=False)).astype(float)

    g = df.groupby('fighter', sort=False)

    # Number of fights BEFORE this one (0-indexed cumcount)
    df['cum_fights'] = g.cumcount()

    # Cumulative sums shifted — subtract current row to get pre-fight total
    for src, dst in [('won', '_cs_won'), ('_ko', '_cs_ko'), ('_sub', '_cs_sub'), ('_fin', '_cs_fin')]:
        df[dst] = g[src].cumsum() - df[src]

    safe_n = df['cum_fights'].clip(lower=1)
    df['career_win_rate']    = np.where(df['cum_fights'] > 0, df['_cs_won'] / safe_n, 0.5)
    df['ko_finish_rate']     = np.where(df['cum_fights'] > 0, df['_cs_ko']  / safe_n, 0.0)
    df['sub_finish_rate']    = np.where(df['cum_fights'] > 0, df['_cs_sub'] / safe_n, 0.0)
    df['career_finish_rate'] = np.where(df['cum_fights'] > 0, df['_cs_fin'] / safe_n, 0.0)

    # Rolling-window stats: shift(1) within group then roll
    def _roll(series, w, default):
        return series.shift(1).rolling(w, min_periods=1).mean().fillna(default)

    df['last3_win_rate']    = g['won'].transform(lambda x: _roll(x, 3,  0.5))
    df['last10_win_rate']   = g['won'].transform(lambda x: _roll(x, 10, 0.5))
    df['last5_won']         = g['won'].transform(lambda x: _roll(x, 5,  0.5))
    df['last5_finish_rate'] = g['_fin'].transform(lambda x: _roll(x, 5, 0.0))
    df['trend_score']       = df['last3_win_rate'] - df['last10_win_rate']

    # Layoff days (days since previous fight per fighter)
    df['_prev_date'] = g['date'].transform(lambda x: x.shift(1))
    df['layoff_days'] = (df['date'] - df['_prev_date']).dt.days.fillna(180.0).clip(lower=0)

    # Opponent quality — loop required (variable-length lookback over names)
    opp_col    = df['opponent'].tolist()
    fighter_col = df['fighter'].tolist()
    idx_list   = df.index.tolist()

    # Build a position map: fighter -> sorted list of row indices
    from collections import defaultdict
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

    # Drop helper columns
    df.drop(columns=['_ko','_sub','_fin','_cs_won','_cs_ko','_cs_sub','_cs_fin','_prev_date'],
            inplace=True)

    return df[['fighter', 'date', 'cum_fights', 'career_win_rate',
               'ko_finish_rate', 'sub_finish_rate', 'career_finish_rate',
               'last3_win_rate', 'last10_win_rate', 'last5_won',
               'last5_finish_rate', 'trend_score', 'layoff_days', 'opp_quality']]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _layoff_buckets(prefix, days_series):
    d = days_series.fillna(180.0)
    return {
        f'{prefix}layoff_lt90':    (d < 90).astype(int),
        f'{prefix}layoff_90_180':  ((d >= 90)  & (d < 180)).astype(int),
        f'{prefix}layoff_180_365': ((d >= 180) & (d < 365)).astype(int),
        f'{prefix}layoff_gt365':   (d >= 365).astype(int),
    }


def corner_flip(X, y):
    """Double training set by swapping Red ↔ Blue and negating diffs."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def compute_weights(dates, cutoff=pd.Timestamp('2024-01-01'), half_life_days=730):
    days_before = (cutoff - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_before / half_life_days)


def compute_qa_stats(career_df, elo_hist_df):
    """Opponent-elo-weighted QA stats. shift(1) — no leakage."""
    from collections import defaultdict
    elo_lookup = elo_hist_df[['fighter', 'date', 'elo_before']].copy()
    opp_elo_df = career_df[['fighter', 'opponent', 'date', 'won', 'got_finish']].copy()
    opp_elo_df = opp_elo_df.rename(columns={'opponent': 'opp_name'})
    opp_ref = elo_lookup.rename(columns={'fighter': 'opp_name', 'elo_before': 'opp_elo'})
    opp_elo_df = pd.merge_asof(opp_elo_df.sort_values('date'),
                                opp_ref.sort_values('date'),
                                on='date', by='opp_name', direction='backward')
    opp_elo_df['opp_elo'] = opp_elo_df['opp_elo'].fillna(1500.0)
    opp_elo_df['ew']      = opp_elo_df['opp_elo'] / 1500.0

    qa_rows = []
    for fighter, grp in opp_elo_df.groupby('fighter', sort=False):
        grp = grp.sort_values('date')
        n = len(grp)
        qa_wr, qa_fr, qa_sl, qa_sa = (np.full(n, 0.5), np.full(n, 0.0),
                                       np.full(n, 0.0), np.full(n, 0.0))
        cum_ew = cum_eww = cum_ewf = 0.0
        cum_n = cum_off = cum_def = 0.0
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
    return pd.concat(qa_rows, ignore_index=True).sort_values(['fighter', 'date'])


def compute_interaction_features(df, career_df):
    """got_finished_rate + age_x_layoff + finish_danger_mismatch."""
    cdf = career_df.sort_values(['fighter', 'date']).copy()
    cdf['is_loss']    = (cdf['won'] == 0).astype(float)
    cdf['is_fin_loss'] = ((cdf['won'] == 0) & (cdf['got_finish'].fillna(0) == 1)).astype(float)
    g = cdf.groupby('fighter', sort=False)
    cdf['_cs_l']  = g['is_loss'].cumsum()    - cdf['is_loss']
    cdf['_cs_fl'] = g['is_fin_loss'].cumsum() - cdf['is_fin_loss']
    cdf['got_finished_rate'] = np.where(
        cdf['_cs_l'] > 0, cdf['_cs_fl'] / cdf['_cs_l'], 0.5)
    chin = cdf[['fighter', 'date', 'got_finished_rate']].sort_values(['fighter', 'date'])

    cr = chin.rename(columns={'fighter': 'R_fighter', 'got_finished_rate': 'R_got_finished_rate'})
    cb = chin.rename(columns={'fighter': 'B_fighter', 'got_finished_rate': 'B_got_finished_rate'})
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


def main():
    print('=' * 62)
    print('  8SI UFC Predictor — Model 1 Trainer (Variant V2)')
    print(f'  Blend: {int(LR_WEIGHT*100)}% LR + {int(XGB_WEIGHT*100)}% XGB  |  {len(FEAT_114)} features')
    print(f'  Scope: men\'s fights only | Window: {TRAIN_START}+ | HL={HL_DAYS}d')
    print('=' * 62)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print('\n[1/9] Loading data...')
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

    print(f'   ufc-master: {len(df):,} rows | career: {len(career_df):,} rows | style: {len(style_df):,} rows')

    # ── 2. Elo ────────────────────────────────────────────────────────────────
    print('\n[2/9] Computing Elo (K=48, base=1500, all-time fights)...')
    elo_hist_df, elo_curr_df = compute_elo(df, K=48, base=1500.0)
    print(f'   History rows: {len(elo_hist_df):,} | Unique fighters: {elo_curr_df["fighter"].nunique():,}')

    # ── 3. Career stats ───────────────────────────────────────────────────────
    print('\n[3/9] Computing career stats with shift(1) — no leakage...')
    all_win_rates = {
        f: grp['won'].sum() / max(1, len(grp))
        for f, grp in career_df.groupby('fighter')
    }
    career_stats = compute_career_stats(career_df, all_win_rates)
    print(f'   Career stat rows: {len(career_stats):,}')

    # ── 4. QA stats ───────────────────────────────────────────────────────────
    print('\n[4/9] Computing opponent-quality-adjusted stats...')
    qa_stats = compute_qa_stats(career_df, elo_hist_df)
    print(f'   QA stat rows: {len(qa_stats):,} ({qa_stats["fighter"].nunique():,} fighters)')

    # ── 5. Filter to training window + men's only ─────────────────────────────
    print(f'\n[5/9] Filtering master to {TRAIN_START}+ and men\'s fights only...')
    df = df[df['date'] >= TRAIN_START].copy()
    df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
    n_before_mens = len(df)
    df = df[~df['weight_class'].isin(WOMENS_CLASSES)].copy()
    df = df.sort_values('date').reset_index(drop=True)
    print(f'   Women\'s fights excluded: {n_before_mens - len(df):,} | Remaining: {len(df):,}')

    # ── 6. Merge all stats ────────────────────────────────────────────────────
    print('\n[6/9] Merging career stats, QA stats, and Elo onto fight data...')
    career_stats = career_stats.sort_values(['fighter', 'date'])
    career_cols  = [c for c in career_stats.columns if c not in ('fighter', 'date')]
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

    # Merge QA stats
    qa_r = qa_stats.rename(columns={'fighter': 'R_fighter',
        'qa_win_rate': 'R_qa_win_rate', 'qa_finish_rate': 'R_qa_finish_rate',
        'qa_SLpM': 'R_qa_SLpM', 'qa_SApM': 'R_qa_SApM'})
    qa_b = qa_stats.rename(columns={'fighter': 'B_fighter',
        'qa_win_rate': 'B_qa_win_rate', 'qa_finish_rate': 'B_qa_finish_rate',
        'qa_SLpM': 'B_qa_SLpM', 'qa_SApM': 'B_qa_SApM'})
    df = pd.merge_asof(df.sort_values('date'), qa_r.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), qa_b.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')
    for c in ['R_qa_win_rate', 'R_qa_finish_rate', 'R_qa_SLpM', 'R_qa_SApM',
              'B_qa_win_rate', 'B_qa_finish_rate', 'B_qa_SLpM', 'B_qa_SApM']:
        df[c] = df[c].fillna(0.5 if 'win_rate' in c else 0.0)
    df['qa_win_rate_dif']    = df['R_qa_win_rate']    - df['B_qa_win_rate']
    df['qa_finish_rate_dif'] = df['R_qa_finish_rate'] - df['B_qa_finish_rate']
    df['qa_SLpM_dif']        = df['R_qa_SLpM']        - df['B_qa_SLpM']
    df['qa_SApM_dif']        = df['R_qa_SApM']        - df['B_qa_SApM']

    # Merge Elo
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

    # ── 7. Feature engineering ────────────────────────────────────────────────
    print('\n[7/9] Engineering features...')
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

    # Interaction features (got_finished_rate, age_x_layoff, finish_danger_mismatch)
    df = compute_interaction_features(df, career_df)

    # Force all features numeric
    for col in FEAT_114:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    # ── 8. Fight filter ───────────────────────────────────────────────────────
    print('\n[8/9] Applying fight filter (R_cum_fights >= 1 AND B_cum_fights >= 1)...')
    n_before = len(df)
    df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy()
    print(f'   Excluded: {n_before - len(df):,} fights | Kept: {len(df):,}')

    missing = [f for f in FEAT_114 if f not in df.columns]
    if missing:
        raise RuntimeError(f'MISSING {len(missing)} features: {missing}')
    print(f'   ✓ All {len(FEAT_114)} features confirmed present')

    # ── 9. Split, recency weight, augment, train, evaluate, save ─────────────
    print('\n[9/9] Splitting, augmenting, training, evaluating...')

    df['target'] = (df['Winner'] == 'Red').astype(int)
    train_mask = df['date'] < TRAIN_CUTOFF
    test_mask  = df['date'] >= TRAIN_CUTOFF

    X_train_raw = df.loc[train_mask, FEAT_114].reset_index(drop=True)
    y_train_raw = df.loc[train_mask, 'target'].reset_index(drop=True)
    d_train_raw = df.loc[train_mask, 'date'].reset_index(drop=True)
    X_test  = df.loc[test_mask,  FEAT_114].reset_index(drop=True)
    y_test  = df.loc[test_mask,  'target'].reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    print(f'   Train rows (pre-aug): {len(X_train_raw):,} | Test rows: {len(X_test):,}')

    # Recency weights
    w_raw = pd.Series(compute_weights(d_train_raw, half_life_days=HL_DAYS),
                      index=y_train_raw.index)
    X_train_aug, y_train_aug, w_train_aug = corner_flip(X_train_raw, y_train_raw, w_raw)
    print(f'   Train rows (post-aug): {len(X_train_aug):,}  (corner-flip ×2)')
    w_arr = w_train_aug.values

    model_lr = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=0.00711, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    model_lr.fit(X_train_aug, y_train_aug, lr__sample_weight=w_arr)

    model_xgb = XGBClassifier(**XGB_PARAMS)
    model_xgb.fit(X_train_aug, y_train_aug, sample_weight=w_arr)

    prob_lr  = model_lr.predict_proba(X_test)
    prob_xgb = model_xgb.predict_proba(X_test)
    prob     = LR_WEIGHT * prob_lr + XGB_WEIGHT * prob_xgb
    y_pred   = (prob[:, 1] > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)

    print(f'\n   ── Temporal accuracy: {acc:.4f}  ({acc*100:.2f}%) ──')

    df_test['_pred'] = y_pred
    print('\n   Per-year accuracy:')
    for yr, grp in df_test.groupby(df_test['date'].dt.year):
        yr_acc = accuracy_score(grp['target'], grp['_pred'])
        print(f'     {yr}: {yr_acc:.3f}  ({len(grp):,} fights)')

    print('\n   Classification report:')
    print(classification_report(y_test, y_pred, target_names=['Blue wins', 'Red wins']))

    if not (0.70 <= acc <= 0.76):
        print(f'WARNING: accuracy {acc:.4f} outside expected 70%–76%')

    # ── Save outputs ──────────────────────────────────────────────────────────
    print('\n   Saving outputs...')

    joblib.dump(model_lr,  os.path.join(MODEL, 'ufc_model_best.pkl'))
    joblib.dump(model_xgb, os.path.join(MODEL, 'ufc_model_xgb.pkl'))
    joblib.dump(FEAT_114,  os.path.join(MODEL, 'feature_columns_best.pkl'))

    metadata = {
        'model_type': 'M1_V2_blend_LR70_XGB30_mens',
        'temporal_accuracy': round(float(acc), 6),
        'n_features': len(FEAT_114),
        'feature_list': FEAT_114,
        'blend_ratio': f'{int(LR_WEIGHT*100)}% LR + {int(XGB_WEIGHT*100)}% XGB',
        'training_window': f'{TRAIN_START} to <{TRAIN_CUTOFF}',
        'test_window': f'>= {TRAIN_CUTOFF} (men\'s fights only)',
        'model_scope': "men's fights only (women's excluded — separate model planned)",
        'recency_weighting': f'exponential decay, half-life={HL_DAYS} days',
        'fight_filter': "R_cum_fights >= 1 AND B_cum_fights >= 1, men's weight classes only",
        'xgb_params': XGB_PARAMS,
        'date_trained': datetime.now().isoformat(),
    }
    with open(os.path.join(MODEL, 'model_metadata.json'), 'w') as fh:
        json.dump(metadata, fh, indent=2)

    elo_hist_df.to_csv(os.path.join(DATA, 'elo_ratings_history.csv'), index=False)
    elo_curr_df.to_csv(os.path.join(DATA, 'elo_current.csv'), index=False)

    saved = [
        os.path.join(MODEL, 'ufc_model_best.pkl'),
        os.path.join(MODEL, 'ufc_model_xgb.pkl'),
        os.path.join(MODEL, 'feature_columns_best.pkl'),
        os.path.join(MODEL, 'model_metadata.json'),
        os.path.join(DATA,  'elo_ratings_history.csv'),
        os.path.join(DATA,  'elo_current.csv'),
    ]
    for path in saved:
        size_kb = os.path.getsize(path) / 1024
        print(f'     ✓ {os.path.relpath(path, ROOT):<45}  {size_kb:,.1f} KB')

    print(f'\n{"=" * 62}')
    print(f'  Training complete.')
    print(f'  Temporal accuracy : {acc*100:.2f}%')
    print(f'  Train / Test      : {len(X_train_raw):,} / {len(X_test):,} fights')
    print(f'  Output files      : {len(saved)}')
    print(f'{"=" * 62}\n')


if __name__ == '__main__':
    main()
