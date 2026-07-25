#!/usr/bin/env python3
"""
8SI UFC Predictor — Model 1 Production Trainer
8SI remediation, Phase 1 (leakage fix):
  Style stats (SLpM, SApM, Str_Acc, Str_Def, TD_Avg, TD_Acc, TD_Def, Sub_Avg)
  now come from an as-of merge (training/style_stats.py, sourced from
  ufc_gold_dataset_final.csv) instead of a static current-career-snapshot
  merge — every historical fight previously saw the fighter's FUTURE
  averages. Added R/B_style_missing + R/B_gf_missing indicators, imputed via
  weight-class training-window medians instead of fillna(0.0)/fillna(0.5).
  See docs/REBASELINE.md for the accuracy delta this caused (expected: a
  drop, since real signal was removed, not added).

8SI v2 Stage 0.2:
  Dropped qa_SLpM/qa_SApM/qa_SLpM_dif/qa_SApM_dif (6 columns) — pure
  duplicates of qa_win_rate/qa_win_rate_dif, not new signal (see
  docs/V2_LOG.md). Features: 133 -> 127.

Variant V2 (May 2026 sprint):
  Blend:   70% LogisticRegression + 30% XGBoost
  Features: 127 (109 base + 6 QA stats + 8 interaction features + 4 Phase 1
            missingness indicators)
  Scope:   Men's weight classes only (women's excluded)
  Window:  2015-01-01 to 2023-12-31 (expanded from 2018)
  Recency: Exponential decay, half-life=730 days
  XGB params: Optuna-tuned, 25 trials, 5-fold CV
  Accuracy: see docs/REBASELINE.md (this docstring's number goes stale the
            moment the leakage fix changes it — don't trust a hardcoded
            figure here again)

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
from sklearn.metrics import accuracy_score, classification_report, log_loss, brier_score_loss
from xgboost import XGBClassifier
import joblib
import json

# ─── Paths (resolve relative to this file so script is runnable from anywhere) ─
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(ROOT, 'data')
MODEL = os.path.join(ROOT, 'model')

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from training.style_stats import compute_style_stats_asof
from features.constants import WC_ORDER, WOMENS_CLASSES, layoff_bucket_flags_vectorized

TRAIN_START  = '2015-01-01'   # expanded from 2018 — data quality check passed
TRAIN_CUTOFF = '2024-01-01'   # train on [START, CUTOFF), test on [CUTOFF, ∞)
LR_WEIGHT    = 0.70
XGB_WEIGHT   = 0.30
HL_DAYS      = 730            # recency half-life in days (2 years)

# Bump ONLY when the feature LIST (names/count) changes — not when a
# feature's formula changes without adding/removing/renaming a column.
# backend/main.py asserts this against model_metadata.json at load time so
# a schema-incompatible model deployed to the serving paths fails loudly
# instead of silently producing garbage predictions.
FEATURE_SCHEMA_VERSION = 3   # v1 = pre-8SI-remediation (129 features), v2 = 8SI Phases 1-2 (133 features), v3 = 8SI v2 Stage 0.2 (127 features, dropped duplicate qa_SLpM/qa_SApM)

# 8SI Phase 7: promoted Elo enhancements (see experiments/elo_v2/ and
# docs/DECISIONS.md for the walk-forward evaluation gating this). Does NOT
# bump FEATURE_SCHEMA_VERSION — same column names/count, formula only.
# Phase 7's original K-sensitivity grid {24,32,40,48} found K=48 optimal
# in that range; method-weighted K + 25% layoff regression together
# measurably improved pooled walk-forward log loss (0.6200 -> 0.6152).
#
# 8SI v2 Stage 0.3 revisited K with a wider grid and found K=48 was NOT
# actually optimal — the lone-feature elo_dif LR diagnostic kept improving
# monotonically all the way to K~192-256 before reversing, but that's a
# proxy metric, not the thing that matters (see the v2 spec's own
# north-star rule: ships only on FULL MODEL pooled walk-forward log loss
# improvement). Checking the actual full-model metric across the
# diagnostic's suggested range found a genuine, non-monotonic local
# optimum at K=96 (0.6166 -> 0.6152, confirmed against neighbors K=80
# and K=112, both worse) — the lone-feature grid's global suggestion
# (K~192-256) was actually worse for the full model (0.6165, 0.6175).
# See docs/V2_LOG.md for the full grid.
ELO_K = 96
ELO_METHOD_MULTIPLIERS = {'KO/TKO': 1.25, 'SUB': 1.25, 'S-DEC': 0.75}
ELO_LAYOFF_REGRESSION = (365, 0.25)   # (days_threshold, regress_pct)

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
    "R_style_missing", "B_style_missing",
    "SLpM_dif", "SApM_dif", "Str_Def_dif", "TD_Def_dif", "Sub_Avg_dif", "TD_Avg_dif",
    "R_elo", "B_elo", "elo_dif",
    "R_elo_trend", "B_elo_trend", "elo_trend_dif",
]

# QA stats: opponent-elo-weighted career metrics (higher signal than raw).
# qa_SLpM/qa_SApM dropped (8SI v2 Stage 0.2) — qa_SLpM == qa_win_rate and
# qa_SApM == 1 - qa_win_rate exactly (compute_qa_stats() never actually
# consumed striking-volume data), so those 6 columns were pure duplicates,
# not new signal. See docs/V2_LOG.md.
FEAT_QA = [
    "R_qa_win_rate", "R_qa_finish_rate",
    "B_qa_win_rate", "B_qa_finish_rate",
    "qa_win_rate_dif", "qa_finish_rate_dif",
]

# Interaction features: age×layoff, finishing danger, chin proxy
FEAT_INT = [
    "R_age_x_layoff", "B_age_x_layoff", "age_x_layoff_dif",
    "R_finish_danger", "B_finish_danger", "finish_danger_mismatch",
    "R_got_finished_rate", "B_got_finished_rate",
    "R_gf_missing", "B_gf_missing",
]

FEAT_114 = FEAT_BASE + FEAT_QA + FEAT_INT   # variable kept for compatibility

# Optuna-tuned XGB params (25 trials, 5-fold CV, May 2026 sprint)
XGB_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6,
    'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'gamma': 0.3, 'reg_alpha': 0, 'reg_lambda': 2.0,
    'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Elo computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_elo(df_all, K=48, base=1500.0, method_multipliers=None, layoff_regression=None):
    """
    Compute Elo from ufc-master.csv (one row per fight, both corners).

    method_multipliers, layoff_regression are experimental, OFF by default
    (None) — with both None this function is byte-for-byte identical to
    its pre-Phase-7 behavior. See experiments/elo_v2/ (8si_remediation_plan.md
    Phase 7) for the walk-forward evaluation gating their promotion.

    method_multipliers: optional dict mapping df_all['finish'] values (e.g.
      'KO/TKO', 'SUB', 'S-DEC') to a K multiplier for that fight. Values not
      in the dict get multiplier 1.0. Requires a 'finish' column in df_all.
    layoff_regression: optional (days_threshold, regress_pct) tuple. A
      fighter's PRE-fight rating is regressed toward `base` by
      regress_pct if their gap since their last recorded fight exceeds
      days_threshold — modeling increased uncertainty after a long layoff.

    Returns:
        history_df  — fighter, opponent, date, elo_before, elo_after, result, elo_trend
        current_df  — fighter, current_elo, last_fight_date, total_fights
    """
    df_sorted = df_all.sort_values('date').reset_index(drop=True)
    elo = {}
    last_fight_date = {}
    history_rows = []

    for _, row in df_sorted.iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        date, winner = row['date'], row['Winner']

        r_before = elo.get(r, base)
        b_before = elo.get(b, base)

        if layoff_regression is not None:
            days_threshold, regress_pct = layoff_regression
            if r in last_fight_date and (date - last_fight_date[r]).days > days_threshold:
                r_before = r_before + regress_pct * (base - r_before)
            if b in last_fight_date and (date - last_fight_date[b]).days > days_threshold:
                b_before = b_before + regress_pct * (base - b_before)

        r_exp = 1.0 / (1.0 + 10.0 ** ((b_before - r_before) / 400.0))
        b_exp = 1.0 - r_exp

        if winner == 'Red':
            r_act, b_act = 1.0, 0.0
        elif winner == 'Blue':
            r_act, b_act = 0.0, 1.0
        else:
            r_act, b_act = 0.5, 0.5

        k_mult = 1.0
        if method_multipliers is not None:
            k_mult = method_multipliers.get(row.get('finish'), 1.0)

        r_after = r_before + K * k_mult * (r_act - r_exp)
        b_after = b_before + K * k_mult * (b_act - b_exp)

        history_rows.append({'fighter': r, 'opponent': b, 'date': date,
                              'elo_before': r_before, 'elo_after': r_after, 'result': r_act})
        history_rows.append({'fighter': b, 'opponent': r, 'date': date,
                              'elo_before': b_before, 'elo_after': b_after, 'result': b_act})

        elo[r] = r_after
        elo[b] = b_after
        last_fight_date[r] = date
        last_fight_date[b] = date

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
def compute_career_stats(career_df):
    """
    Per-fighter career stats computed with shift(1) — every stat row
    represents what was known BEFORE that fight.

    Uses vectorized cumsum-shift for rates and a tight loop for
    rolling-window and opp_quality stats. opp_quality scores each of a
    fighter's last <=5 opponents by that opponent's OWN career_win_rate
    as of THIS fight's date (not their full-career rate, which would leak
    fights that hadn't happened yet) — see 8si_remediation_plan.md Phase 2.
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
    opp_col     = df['opponent'].tolist()
    fighter_col = df['fighter'].tolist()
    date_col    = df['date'].tolist()
    idx_list    = df.index.tolist()

    # POST-fight cumulative win rate (includes this row's own outcome) —
    # deliberately NOT career_win_rate, which is shift(1)/pre-fight and
    # would exclude the outcome of an opponent's most recent fight before
    # the date being evaluated, undercounting their as-of win rate by one
    # fight.
    post_win_rate = ((df['_cs_won'] + df['won']) / (df['cum_fights'] + 1)).tolist()

    # Build a position map: fighter -> sorted list of row indices. df is
    # already sorted by ['fighter', 'date'], so each fighter's positions are
    # date-ordered — this doubles as an as-of lookup table (date_col[p],
    # post_win_rate[p]) of that fighter's OWN cumulative win rate through
    # each of their own fights.
    from collections import defaultdict
    fighter_positions = defaultdict(list)
    for pos, idx in enumerate(idx_list):
        fighter_positions[fighter_col[pos]].append(pos)

    asof_dates = {f: np.array([date_col[p] for p in positions])
                  for f, positions in fighter_positions.items()}
    asof_rates = {f: np.array([post_win_rate[p] for p in positions])
                  for f, positions in fighter_positions.items()}

    def _win_rate_asof(opp, date):
        """opp's cumulative win rate through their last fight strictly
        before `date`, or None if opp has no such fight (unknown opponent,
        or opp's first-ever fight)."""
        dates = asof_dates.get(opp)
        if dates is None or len(dates) == 0:
            return None
        i = np.searchsorted(dates, date, side='left') - 1
        return asof_rates[opp][i] if i >= 0 else None

    opp_quality_arr = np.full(len(df), 0.5)
    for fighter, positions in fighter_positions.items():
        for rank, pos in enumerate(positions):
            past_opps = [opp_col[p] for p in positions[max(0, rank - 5):rank]]
            date = date_col[pos]
            rates = [r for r in (_win_rate_asof(opp, date) for opp in past_opps) if r is not None]
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
def _impute_by_weight_class(df, stats, weight_col, train_mask):
    """
    For each stat name in `stats`, impute NaN in R_<stat>/B_<stat> with the
    training-window median of that stat within the row's weight class
    (fallback: training-window median across all weight classes). R- and
    B-corner observations are pooled since a stat's distribution is a
    per-fighter property, not a per-corner one. Mutates df in place.
    """
    r_pooled = df.loc[train_mask, [weight_col] + [f'R_{s}' for s in stats]]
    r_pooled = r_pooled.rename(columns={f'R_{s}': s for s in stats})
    b_pooled = df.loc[train_mask, [weight_col] + [f'B_{s}' for s in stats]]
    b_pooled = b_pooled.rename(columns={f'B_{s}': s for s in stats})
    pooled = pd.concat([r_pooled, b_pooled], ignore_index=True)

    wc_medians = pooled.groupby(weight_col)[stats].median()
    global_medians = pooled[stats].median()

    for prefix in ('R_', 'B_'):
        for s in stats:
            col = f'{prefix}{s}'
            med = df[weight_col].map(wc_medians[s]).fillna(global_medians[s])
            df[col] = df[col].fillna(med)


def _corner_swap(X):
    """Return a copy of X with R_/B_ column pairs swapped and _dif columns
    negated — i.e. relabel the Blue fighter as Red and vice versa. The
    single-frame transform that corner_flip() augments training data with,
    and that predict_symmetric() uses at inference time."""
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
    return Xf


def corner_flip(X, y, w):
    """Double training set by swapping Red ↔ Blue and negating diffs."""
    Xf = _corner_swap(X)
    return (pd.concat([X, Xf], ignore_index=True),
            pd.concat([y, 1 - y], ignore_index=True),
            pd.concat([w, w], ignore_index=True))


def predict_symmetric(model_lr, model_xgb, X, lr_weight, xgb_weight):
    """
    Corner-invariant blended P(Red wins). Corner-flip augmentation at
    training time doesn't make the model perfectly symmetric — it can still
    give a different prediction for the same fight depending on which
    fighter is drawn Red vs. Blue. This averages the direct prediction with
    the complement of the prediction on the corner-swapped input, which
    cancels that bias out: p = 0.5 * (P(Red wins | as given) +
    (1 - P("Red" wins | corners swapped))). Use this (not the raw blend)
    for all test-set evaluation and, eventually, the backend prediction
    path — see 8si_remediation_plan.md Phase 3.1.
    """
    X_swapped = _corner_swap(X)

    def _blend(frame):
        p_lr  = model_lr.predict_proba(frame)[:, 1]
        p_xgb = model_xgb.predict_proba(frame)[:, 1]
        return lr_weight * p_lr + xgb_weight * p_xgb

    p_orig    = _blend(X)
    p_swapped = _blend(X_swapped)
    return 0.5 * (p_orig + (1.0 - p_swapped))


def calibration_table(y_true, y_prob, n_bins=10):
    """
    10 equal-width bins over [0, 1]. For each bin: count, mean predicted
    probability, and observed (actual) win rate — the gap between the two
    is the calibration error in that bin. Returns a list of dicts (JSON-
    serializable, for model_metadata.json).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        rows.append({
            'bin': f'[{edges[b]:.1f}, {edges[b+1]:.1f})',
            'n': n,
            'mean_predicted': round(float(y_prob[mask].mean()), 4) if n > 0 else None,
            'observed_rate': round(float(y_true[mask].mean()), 4) if n > 0 else None,
        })
    return rows


def print_calibration_table(rows, title='Calibration table'):
    print(f'\n   {title} (predicted vs. observed P(Red wins), 10 bins):')
    print(f'     {"Bin":<14}{"N":>7}{"Mean Pred":>12}{"Observed":>12}')
    for r in rows:
        mp = f"{r['mean_predicted']:.4f}" if r['mean_predicted'] is not None else '—'
        ob = f"{r['observed_rate']:.4f}" if r['observed_rate'] is not None else '—'
        print(f"     {r['bin']:<14}{r['n']:>7}{mp:>12}{ob:>12}")


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


def compute_got_finished_rate(career_df):
    """
    Per-fighter, per-fight 'chin' proxy: cumulative rate of prior losses
    that were finishes (KO/TKO/Sub), shift(1) — pre-fight, no leakage.
    NaN where the fighter has no prior loss yet (undefeated, or debut).
    Shared by compute_interaction_features() (trainer) and backend/main.py's
    as-of-today lookup (8SI Phase 5 Part A) — a single source of truth so
    the two can't silently diverge.
    """
    cdf = career_df.sort_values(['fighter', 'date']).copy()
    cdf['is_loss']     = (cdf['won'] == 0).astype(float)
    cdf['is_fin_loss'] = ((cdf['won'] == 0) & (cdf['got_finish'].fillna(0) == 1)).astype(float)
    g = cdf.groupby('fighter', sort=False)
    cdf['_cs_l']  = g['is_loss'].cumsum()    - cdf['is_loss']
    cdf['_cs_fl'] = g['is_fin_loss'].cumsum() - cdf['is_fin_loss']
    cdf['got_finished_rate'] = np.where(
        cdf['_cs_l'] > 0, cdf['_cs_fl'] / cdf['_cs_l'], np.nan)
    return cdf[['fighter', 'date', 'got_finished_rate']].sort_values(['fighter', 'date'])


def compute_interaction_features(df, career_df, train_cutoff):
    """got_finished_rate + age_x_layoff + finish_danger_mismatch."""
    chin = compute_got_finished_rate(career_df)

    cr = chin.rename(columns={'fighter': 'R_fighter', 'got_finished_rate': 'R_got_finished_rate'})
    cb = chin.rename(columns={'fighter': 'B_fighter', 'got_finished_rate': 'B_got_finished_rate'})
    df = pd.merge_asof(df.sort_values('date'), cr.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), cb.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')
    # NaN means "no prior loss to be finished in yet" (undefeated fighter or
    # debut) — impute to weight-class median rather than a hardcoded 0.5.
    df['R_gf_missing'] = df['R_got_finished_rate'].isna().astype(int)
    df['B_gf_missing'] = df['B_got_finished_rate'].isna().astype(int)
    train_mask = df['date'] < train_cutoff
    _impute_by_weight_class(df, ['got_finished_rate'], 'weight_class', train_mask)

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


class PlattCalibrator:
    """
    Platt/sigmoid probability calibrator: a 2-parameter logistic fit of
    observed outcome on raw predicted probability. Deliberately NOT
    IsotonicRegression — isotonic's many-knot step function needs far more
    calibration data than we have (~600 rows) to avoid overfitting the
    validation window's noise; both were tried empirically here (see
    docs/REBASELINE.md Phase 3) and isotonic was worse. sklearn's own docs
    make the same recommendation: prefer Platt/sigmoid when calibration
    data is limited. Exposes .predict(p) -> calibrated probability array,
    same call shape IsotonicRegression would have had, so this is a
    drop-in if a future retrain has enough data to revisit that choice.
    """
    def __init__(self):
        self._lr = LogisticRegression()

    def fit(self, p, y):
        self._lr.fit(np.asarray(p).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, p):
        return self._lr.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def fit_calibrator(df, train_start, train_cutoff, calib_months=18):
    """
    Fits a probability calibrator (see PlattCalibrator) mapping raw
    symmetric blended probability -> observed win rate. To avoid fitting it
    on predictions the deployed model has already memorized, a SEPARATE
    pair of blend models is trained on data strictly before a held-out
    validation window (the last `calib_months` months of the training
    period, excluded from that fitting), and the calibrator learns from
    THEIR out-of-sample predictions on that window. The deployed models
    (trained on the full window in main()) are unaffected — only the
    calibrator artifact is a function of this fit. Returns
    (calibrator_or_None, info_dict).
    """
    calib_start = pd.Timestamp(train_cutoff) - pd.DateOffset(months=calib_months)
    calib_fit_mask = df['date'] < calib_start
    calib_val_mask = (df['date'] >= calib_start) & (df['date'] < train_cutoff)
    n_val = int(calib_val_mask.sum())

    if calib_fit_mask.sum() < 100 or n_val < 30:
        return None, {'calib_window': None, 'n_calib_val': n_val,
                       'note': 'insufficient data for calibration fit, skipped'}

    X_fit_raw = df.loc[calib_fit_mask, FEAT_114].reset_index(drop=True)
    y_fit_raw = df.loc[calib_fit_mask, 'target'].reset_index(drop=True)
    d_fit_raw = df.loc[calib_fit_mask, 'date'].reset_index(drop=True)
    w_fit_raw = pd.Series(compute_weights(d_fit_raw, half_life_days=HL_DAYS), index=y_fit_raw.index)
    X_fit_aug, y_fit_aug, w_fit_aug = corner_flip(X_fit_raw, y_fit_raw, w_fit_raw)

    cal_lr = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=0.00711, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    cal_lr.fit(X_fit_aug, y_fit_aug, lr__sample_weight=w_fit_aug.values)
    cal_xgb = XGBClassifier(**XGB_PARAMS)
    cal_xgb.fit(X_fit_aug, y_fit_aug, sample_weight=w_fit_aug.values)

    X_val = df.loc[calib_val_mask, FEAT_114].reset_index(drop=True)
    y_val = df.loc[calib_val_mask, 'target'].reset_index(drop=True)
    p_val = predict_symmetric(cal_lr, cal_xgb, X_val, LR_WEIGHT, XGB_WEIGHT)

    calibrator = PlattCalibrator().fit(p_val, y_val)

    info = {
        'calib_window': f'{calib_start.date()} to <{train_cutoff}',
        'n_calib_val': n_val,
        'method': 'platt_sigmoid',
    }
    return calibrator, info


def build_dataset(train_start, train_cutoff, data_dir, elo_kwargs=None):
    """
    Runs the full feature-engineering pipeline (load, Elo, career stats, QA
    stats, style stats, feature engineering, fight filter) and returns
    (df, elo_hist_df, elo_curr_df). `df` has 'date', 'target' (via Winner,
    added by the caller — NOT set here), and every FEAT_114 column, filtered
    to R_cum_fights >= 1 and B_cum_fights >= 1. Train/test split is the
    caller's job (this function doesn't know train_cutoff's role beyond
    feeding it to interaction-feature and style-stat median imputation,
    which use ONLY the [train_start, train_cutoff) window — so a caller
    must NOT reuse the same build_dataset() result across different
    train_cutoff values without leakage; call it fresh per fold. Shared by
    main() and training/walk_forward.py.

    elo_kwargs: optional dict of extra kwargs passed to compute_elo() (K,
    method_multipliers, layoff_regression) — for Elo-variant experiments
    (see experiments/elo_v2/). None (default, i.e. omitted by the caller)
    uses the promoted ELO_K/ELO_METHOD_MULTIPLIERS/ELO_LAYOFF_REGRESSION
    config. Pass elo_kwargs={} explicitly to get the original pre-Phase-7
    plain K=48 baseline instead (as experiments/elo_v2/run_experiment.py
    does for its own baseline).
    """
    # Distinguish "caller omitted elo_kwargs" (None -> promoted default)
    # from "caller explicitly passed {}" (plain pre-Phase-7 baseline) — an
    # empty dict is falsy, so `elo_kwargs or {...}` would NOT tell those
    # apart; must check `is None` explicitly.
    if elo_kwargs is None:
        elo_kwargs = {'K': ELO_K, 'method_multipliers': ELO_METHOD_MULTIPLIERS, 'layoff_regression': ELO_LAYOFF_REGRESSION}
    else:
        elo_kwargs = dict(elo_kwargs)
    elo_k = elo_kwargs.pop('K', 48)
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print('\n[1/9] Loading data...')
    df = pd.read_csv(os.path.join(data_dir, 'ufc-master.csv'), low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    career_df = pd.read_csv(os.path.join(data_dir, 'career_fights_updated.csv'))
    career_df['date'] = pd.to_datetime(career_df['date'])
    career_df = career_df.sort_values(['fighter', 'date']).reset_index(drop=True)

    gold_df = pd.read_csv(os.path.join(data_dir, 'ufc_gold_dataset_final.csv'))

    print(f'   ufc-master: {len(df):,} rows | career: {len(career_df):,} rows | gold (style): {len(gold_df):,} rows')

    # ── 2. Elo ────────────────────────────────────────────────────────────────
    print('\n[2/9] Computing Elo (K=48, base=1500, all-time fights)...')
    elo_hist_df, elo_curr_df = compute_elo(df, K=elo_k, base=1500.0, **elo_kwargs)
    print(f'   History rows: {len(elo_hist_df):,} | Unique fighters: {elo_curr_df["fighter"].nunique():,}')

    # ── 3. Career stats ───────────────────────────────────────────────────────
    print('\n[3/9] Computing career stats with shift(1) — no leakage...')
    career_stats = compute_career_stats(career_df)
    print(f'   Career stat rows: {len(career_stats):,}')

    # ── 4. QA stats ───────────────────────────────────────────────────────────
    print('\n[4/9] Computing opponent-quality-adjusted stats...')
    qa_stats = compute_qa_stats(career_df, elo_hist_df)
    print(f'   QA stat rows: {len(qa_stats):,} ({qa_stats["fighter"].nunique():,} fighters)')

    # ── 5. Filter to training window + men's only ─────────────────────────────
    print(f'\n[5/9] Filtering master to {train_start}+ and men\'s fights only...')
    df = df[df['date'] >= train_start].copy()
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

    # Merge QA stats — qa_win_rate/qa_finish_rate only (8SI v2 Stage 0.2):
    # qa_SLpM/qa_SApM are dropped here. compute_qa_stats() itself is
    # untouched (still computes them, still used by features/build.py's
    # get_fighter_state_asof() as a general lookup) — this is specifically
    # about the model's feature list. See docs/V2_LOG.md: qa_SLpM ==
    # qa_win_rate and qa_SApM == 1 - qa_win_rate exactly (verified
    # numerically in 8SI Phase 5) — the formula never actually consumed
    # striking-volume data, so these 6 columns (R/B_qa_SLpM, R/B_qa_SApM,
    # qa_SLpM_dif, qa_SApM_dif) were pure duplicates of qa_win_rate/
    # qa_win_rate_dif, not new signal.
    qa_r = qa_stats[['fighter', 'date', 'qa_win_rate', 'qa_finish_rate']].rename(
        columns={'fighter': 'R_fighter', 'qa_win_rate': 'R_qa_win_rate', 'qa_finish_rate': 'R_qa_finish_rate'})
    qa_b = qa_stats[['fighter', 'date', 'qa_win_rate', 'qa_finish_rate']].rename(
        columns={'fighter': 'B_fighter', 'qa_win_rate': 'B_qa_win_rate', 'qa_finish_rate': 'B_qa_finish_rate'})
    df = pd.merge_asof(df.sort_values('date'), qa_r.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), qa_b.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')
    for c in ['R_qa_win_rate', 'R_qa_finish_rate', 'B_qa_win_rate', 'B_qa_finish_rate']:
        df[c] = df[c].fillna(0.5 if 'win_rate' in c else 0.0)
    df['qa_win_rate_dif']    = df['R_qa_win_rate']    - df['B_qa_win_rate']
    df['qa_finish_rate_dif'] = df['R_qa_finish_rate'] - df['B_qa_finish_rate']

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

    # Merge style stats — as-of (leak-free), see training/style_stats.py.
    # NaN where the fighter has no prior fight in ufc_gold_dataset_final.csv
    # (their UFC debut, or a name not present in that source); imputed below
    # via _impute_by_weight_class rather than a blind fillna(0.0), since 0
    # SLpM means "worst fighter alive," not "unknown."
    style_asof = compute_style_stats_asof(gold_df)
    style_src = ['SLpM', 'SApM', 'Str_Acc', 'Str_Def', 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']
    r_style = style_asof.rename(columns={'fighter': 'R_fighter', **{c: f'R_{c}' for c in style_src}})
    b_style = style_asof.rename(columns={'fighter': 'B_fighter', **{c: f'B_{c}' for c in style_src}})
    df = pd.merge_asof(df.sort_values('date'), r_style.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b_style.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')
    df['R_style_missing'] = df['R_SLpM'].isna().astype(int)
    df['B_style_missing'] = df['B_SLpM'].isna().astype(int)

    train_window_mask = df['date'] < train_cutoff
    _impute_by_weight_class(df, style_src, 'weight_class', train_window_mask)

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
    for lb in layoff_bucket_flags_vectorized('R_', df['R_layoff_days']).items():
        df[lb[0]] = lb[1].values
    for lb in layoff_bucket_flags_vectorized('B_', df['B_layoff_days']).items():
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
    df = compute_interaction_features(df, career_df, train_cutoff)

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

    return df, elo_hist_df, elo_curr_df


def main(train_start=TRAIN_START, train_cutoff=TRAIN_CUTOFF, data_dir=DATA, output_dir=None):
    # output_dir=None preserves production behavior exactly: model artifacts
    # go to MODEL, generated data (Elo CSVs) go to data_dir. Passing output_dir
    # (e.g. from tests) redirects ALL generated artifacts there instead, so a
    # test run never touches real model/ or data/ outputs.
    model_dir = output_dir if output_dir is not None else MODEL
    elo_dir   = output_dir if output_dir is not None else data_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    print('=' * 62)
    print('  8SI UFC Predictor — Model 1 Trainer (Variant V2)')
    print(f'  Blend: {int(LR_WEIGHT*100)}% LR + {int(XGB_WEIGHT*100)}% XGB  |  {len(FEAT_114)} features')
    print(f'  Scope: men\'s fights only | Window: {train_start}+ | HL={HL_DAYS}d')
    print('=' * 62)

    df, elo_hist_df, elo_curr_df = build_dataset(train_start, train_cutoff, data_dir)

    # ── 9. Split, recency weight, augment, train, evaluate, save ─────────────
    print('\n[9/9] Splitting, augmenting, training, evaluating...')

    df['target'] = (df['Winner'] == 'Red').astype(int)
    train_mask = df['date'] < train_cutoff
    test_mask  = df['date'] >= train_cutoff

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

    # Symmetric (corner-invariant) probabilities for ALL test-set evaluation
    # — see predict_symmetric()'s docstring and 8si_remediation_plan.md
    # Phase 3.1. Threshold at 0.5 on this, not the raw blend.
    p_test = predict_symmetric(model_lr, model_xgb, X_test, LR_WEIGHT, XGB_WEIGHT)
    y_pred = (p_test > 0.5).astype(int)
    acc    = accuracy_score(y_test, y_pred)
    ll     = log_loss(y_test, p_test)
    brier  = brier_score_loss(y_test, p_test)
    calib_rows = calibration_table(y_test, p_test)

    print(f'\n   ── Temporal accuracy: {acc:.4f}  ({acc*100:.2f}%)  |  Log loss: {ll:.4f}  |  Brier: {brier:.4f} ──')
    print_calibration_table(calib_rows)

    # Baselines — the model has to beat these, not 50%.
    elo_pred = (df_test['R_elo'] > df_test['B_elo']).astype(int)
    elo_acc  = float(accuracy_score(y_test, elo_pred))

    fav_acc, fav_coverage = None, 0.0
    if {'R_odds', 'B_odds'}.issubset(df_test.columns):
        odds_ok = df_test[['R_odds', 'B_odds']].notna().all(axis=1)
        fav_coverage = float(odds_ok.mean())
        if odds_ok.sum() > 0:
            # American odds: more negative = bigger favorite.
            fav_pred = (df_test.loc[odds_ok, 'R_odds'] < df_test.loc[odds_ok, 'B_odds']).astype(int)
            fav_acc = float(accuracy_score(y_test[odds_ok.values], fav_pred))

    print('\n   Baselines:')
    print(f'     Pick higher Elo         : {elo_acc:.4f}  ({elo_acc*100:.2f}%)')
    if fav_acc is not None:
        print(f'     Always pick favorite    : {fav_acc:.4f}  ({fav_acc*100:.2f}%)  [odds coverage: {fav_coverage*100:.1f}%]')
    else:
        print('     Always pick favorite    : n/a (no odds coverage in test set)')

    # Probability calibration (Phase 3.4) — fit on a held-out validation
    # window inside the training period; does NOT affect the deployed
    # model_lr/model_xgb above, only the separate calibrator.pkl artifact.
    calibrator, calib_info = fit_calibrator(df, train_start, train_cutoff)
    ll_cal = brier_cal = None
    if calibrator is not None:
        p_test_cal = calibrator.predict(p_test)
        ll_cal    = float(log_loss(y_test, p_test_cal))
        brier_cal = float(brier_score_loss(y_test, p_test_cal))
        print(f'\n   Calibrated ({calib_info["method"]}, fit on {calib_info["calib_window"]}, n={calib_info["n_calib_val"]}):')
        print(f'     Log loss: {ll_cal:.4f}  (raw: {ll:.4f})  |  Brier: {brier_cal:.4f}  (raw: {brier:.4f})')
    else:
        print(f'\n   Calibrator not fit: {calib_info.get("note", "unknown reason")}')

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

    joblib.dump(model_lr,  os.path.join(model_dir, 'ufc_model_best.pkl'))
    joblib.dump(model_xgb, os.path.join(model_dir, 'ufc_model_xgb.pkl'))
    joblib.dump(FEAT_114,  os.path.join(model_dir, 'feature_columns_best.pkl'))
    if calibrator is not None:
        joblib.dump(calibrator, os.path.join(model_dir, 'calibrator.pkl'))

    metadata = {
        'model_type': 'M1_V2_blend_LR70_XGB30_mens',
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'temporal_accuracy': round(float(acc), 6),
        'temporal_log_loss': round(float(ll), 6),
        'temporal_brier': round(float(brier), 6),
        'calibration_table': calib_rows,
        'baseline_higher_elo_accuracy': round(elo_acc, 6),
        'baseline_favorite_accuracy': round(fav_acc, 6) if fav_acc is not None else None,
        'baseline_favorite_odds_coverage': round(fav_coverage, 6),
        'calibrator_fit': calib_info,
        'calibrated_log_loss': round(ll_cal, 6) if ll_cal is not None else None,
        'calibrated_brier': round(brier_cal, 6) if brier_cal is not None else None,
        'probabilities': 'symmetric (predict_symmetric — corner-invariant), not the raw blend',
        'n_features': len(FEAT_114),
        'feature_list': FEAT_114,
        'blend_ratio': f'{int(LR_WEIGHT*100)}% LR + {int(XGB_WEIGHT*100)}% XGB',
        'training_window': f'{train_start} to <{train_cutoff}',
        'test_window': f'>= {train_cutoff} (men\'s fights only)',
        'model_scope': "men's fights only (women's excluded — separate model planned)",
        'recency_weighting': f'exponential decay, half-life={HL_DAYS} days',
        'fight_filter': "R_cum_fights >= 1 AND B_cum_fights >= 1, men's weight classes only",
        'xgb_params': XGB_PARAMS,
        'date_trained': datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as fh:
        json.dump(metadata, fh, indent=2)

    elo_hist_df.to_csv(os.path.join(elo_dir, 'elo_ratings_history.csv'), index=False)
    elo_curr_df.to_csv(os.path.join(elo_dir, 'elo_current.csv'), index=False)

    saved = [
        os.path.join(model_dir, 'ufc_model_best.pkl'),
        os.path.join(model_dir, 'ufc_model_xgb.pkl'),
        os.path.join(model_dir, 'feature_columns_best.pkl'),
        os.path.join(model_dir, 'model_metadata.json'),
        os.path.join(elo_dir,   'elo_ratings_history.csv'),
        os.path.join(elo_dir,   'elo_current.csv'),
    ]
    if calibrator is not None:
        saved.append(os.path.join(model_dir, 'calibrator.pkl'))
    for path in saved:
        size_kb = os.path.getsize(path) / 1024
        print(f'     ✓ {os.path.relpath(path, ROOT):<45}  {size_kb:,.1f} KB')

    print(f'\n{"=" * 62}')
    print(f'  Training complete.')
    print(f'  Temporal accuracy : {acc*100:.2f}%  |  Log loss: {ll:.4f}  |  Brier: {brier:.4f}')
    print(f'  Baselines         : higher-Elo {elo_acc*100:.2f}%'
          + (f' | favorite {fav_acc*100:.2f}%' if fav_acc is not None else ' | favorite n/a'))
    print(f'  Train / Test      : {len(X_train_raw):,} / {len(X_test):,} fights')
    print(f'  Output files      : {len(saved)}')
    print(f'{"=" * 62}\n')

    return acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Model 1 (8SI UFC Predictor)')
    parser.add_argument(
        '--out-dir', default=os.path.join(MODEL, 'v2'),
        help="Directory for ALL generated artifacts (models, feature list, metadata, "
             "Elo CSVs). Defaults to model/v2/ — deliberately NOT the paths "
             "backend/main.py reads (model/ufc_model_best.pkl etc.) — so retraining "
             "never silently overwrites what's being served. Pass the serving paths' "
             "directory explicitly (e.g. --out-dir model) to promote a new model; "
             "that should be a deliberate act, not a side effect of running this script.",
    )
    args = parser.parse_args()
    main(output_dir=args.out_dir)
