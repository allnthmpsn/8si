#!/usr/bin/env python3
"""
Step 2 — Career Trajectory Feature Engineering
Builds new trajectory features from career_fights_updated.csv and joins to master dataset.
Reports correlation of each new feature with Winner_bin.
Saves: experiments/research/augmented_dataset.csv
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
OUT  = os.path.join(ROOT, 'experiments', 'research')

TRAIN_START  = '2018-01-01'
TRAIN_CUTOFF = '2024-01-01'

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

def compute_trajectory_features(career_df):
    """
    Compute trajectory features per fighter per fight (shift=1, no leakage).
    Returns a DataFrame with fighter, date, and 8 trajectory features.
    """
    df = career_df.sort_values(['fighter', 'date']).copy().reset_index(drop=True)

    # Parse DOB from fighter_dob if available; we'll compute age_at_fight from career date and DOB
    # career_fights_updated.csv doesn't have DOB — we'll use a fighters reference file
    # Fallback: approximate from ufc-master.csv R_age + fight date

    # Finish flags
    df['_ko']  = ((df['won'] == 1) & df['method'].str.contains('KO|TKO', case=False, na=False)).astype(float)
    df['_sub'] = ((df['won'] == 1) & df['method'].str.contains('Sub|Submission', case=False, na=False)).astype(float)
    df['_fin'] = ((df['won'] == 1) & df['method'].str.contains('KO|TKO|Sub', case=False, na=False)).astype(float)
    df['_got_fin'] = df['got_finish'].fillna(0).astype(float)  # fighter was finished

    g = df.groupby('fighter', sort=False)
    df['cum_fights'] = g.cumcount()

    # Cumulative career finish rate (pre-fight)
    df['_cs_fin'] = g['_fin'].cumsum() - df['_fin']
    df['_cs_won'] = g['won'].cumsum()  - df['won']
    safe_n = df['cum_fights'].clip(lower=1)
    df['career_finish_rate'] = np.where(df['cum_fights'] > 0, df['_cs_fin'] / safe_n, 0.0)
    df['career_win_rate_traj'] = np.where(df['cum_fights'] > 0, df['_cs_won'] / safe_n, 0.5)

    def _roll(series, w, default):
        return series.shift(1).rolling(w, min_periods=1).mean().fillna(default)

    df['last5_won_traj']         = g['won'].transform(lambda x: _roll(x, 5, 0.5))
    df['last5_finish_rate_traj'] = g['_fin'].transform(lambda x: _roll(x, 5, 0.0))

    # ── 1. fights_since_finish — fights since last KO/TKO win or submission win ──
    def _fights_since_finish(series_fin):
        """How many fights since the fighter last finished someone (pre-fight)."""
        shifted = series_fin.shift(1)
        result = np.zeros(len(shifted))
        count = 0
        for i, v in enumerate(shifted):
            if pd.isna(v):
                result[i] = np.nan
            else:
                if v == 1:
                    count = 0
                else:
                    count += 1
                result[i] = count
        return pd.Series(result, index=series_fin.index)

    df['fights_since_finish'] = g['_fin'].transform(_fights_since_finish)
    df['fights_since_finish'] = df['fights_since_finish'].fillna(0)

    # ── 2. win_rate_last5_vs_career — trend: are they going up or down ──
    df['win_rate_l5_vs_career'] = df['last5_won_traj'] - df['career_win_rate_traj']

    # ── 3. finish_rate_trend — last5 finish rate vs career finish rate ──
    df['finish_rate_trend'] = df['last5_finish_rate_traj'] - df['career_finish_rate']

    # ── 4. longest_losing_streak_ever (pre-fight) ──
    def _max_losing_streak(series_won):
        shifted = series_won.shift(1)
        result  = np.zeros(len(shifted))
        cur_streak = 0
        max_streak = 0
        for i, v in enumerate(shifted):
            if pd.isna(v):
                result[i] = max_streak
            else:
                if v == 0:
                    cur_streak += 1
                    max_streak = max(max_streak, cur_streak)
                else:
                    cur_streak = 0
                result[i] = max_streak
        return pd.Series(result, index=series_won.index)

    df['longest_lose_streak_ever'] = g['won'].transform(_max_losing_streak)

    # ── 5. comeback_flag — won after 2+ fight losing streak in last 10 fights ──
    def _comeback_flag(series_won):
        """1 if fighter won after ≥2-fight losing streak within last 10 fights (pre-fight)."""
        shifted = series_won.shift(1)
        result  = np.zeros(len(shifted))
        for i in range(len(shifted)):
            window = shifted.iloc[max(0, i-10):i].dropna().tolist()
            if not window:
                result[i] = 0
                continue
            streak = 0
            had_streak = False
            comeback = 0
            for j, v in enumerate(window):
                if v == 0:
                    streak += 1
                else:
                    if streak >= 2:
                        had_streak = True
                        comeback = 1
                    streak = 0
            result[i] = int(had_streak and comeback)
        return pd.Series(result, index=series_won.index)

    df['comeback_flag'] = g['won'].transform(_comeback_flag)

    # ── 6. time_between_losses — average fights between losses (consistency proxy) ──
    def _avg_fights_between_losses(series_won):
        shifted = series_won.shift(1).tolist()
        result  = []
        gaps = []
        gap  = 0
        for v in shifted:
            if pd.isna(v):
                result.append(np.nan)
                continue
            if v == 1:
                gap += 1
            else:
                if gaps:
                    gaps.append(gap)
                gap = 0
            result.append(float(np.mean(gaps)) if gaps else np.nan)
        return pd.Series(result, index=series_won.index).fillna(3.0)

    df['avg_fights_between_losses'] = g['won'].transform(_avg_fights_between_losses)

    # Gather output columns
    traj_cols = [
        'fighter', 'date', 'cum_fights',
        'fights_since_finish',
        'win_rate_l5_vs_career',
        'finish_rate_trend',
        'longest_lose_streak_ever',
        'comeback_flag',
        'avg_fights_between_losses',
    ]
    return df[traj_cols].copy()


def main():
    print('=' * 60)
    print('  STEP 2 — Career Trajectory Feature Engineering')
    print('=' * 60)

    print('\n[1] Loading career data...')
    career_df = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_df['date'] = pd.to_datetime(career_df['date'])
    career_df = career_df.sort_values(['fighter', 'date']).reset_index(drop=True)
    print(f'   Career rows: {len(career_df):,}  |  Fighters: {career_df["fighter"].nunique():,}')

    print('\n[2] Computing trajectory features (per-fighter, shift=1, no leakage)...')
    traj = compute_trajectory_features(career_df)
    print(f'   Trajectory rows: {len(traj):,}')

    NEW_FEATURES = [
        'fights_since_finish',
        'win_rate_l5_vs_career',
        'finish_rate_trend',
        'longest_lose_streak_ever',
        'comeback_flag',
        'avg_fights_between_losses',
    ]

    print('\n[3] Loading master data and rebuilding augmented dataset...')
    df_master = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    df_master['date'] = pd.to_datetime(df_master['date'])
    df_master = df_master[df_master['date'] >= TRAIN_START].copy()
    df_master = df_master[df_master['Winner'].isin(['Red', 'Blue'])].copy()
    df_master = df_master.sort_values('date').reset_index(drop=True)

    # Also need age_at_fight: pull from ufc-master R_age/B_age (already there)
    df_master['R_age'] = pd.to_numeric(df_master.get('R_age', 28), errors='coerce').fillna(28.0)
    df_master['B_age'] = pd.to_numeric(df_master.get('B_age', 28), errors='coerce').fillna(28.0)

    # Merge trajectory features for Red corner
    traj_sorted = traj.sort_values(['fighter', 'date'])
    r_traj = traj_sorted.rename(columns={'fighter': 'R_fighter',
                                         **{c: f'R_{c}' for c in NEW_FEATURES + ['cum_fights']}})
    b_traj = traj_sorted.rename(columns={'fighter': 'B_fighter',
                                         **{c: f'B_{c}' for c in NEW_FEATURES + ['cum_fights']}})

    df = pd.merge_asof(df_master.sort_values('date'), r_traj.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b_traj.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')

    # Fill debutants
    defaults = {
        'fights_since_finish': 0.0,
        'win_rate_l5_vs_career': 0.0,
        'finish_rate_trend': 0.0,
        'longest_lose_streak_ever': 0.0,
        'comeback_flag': 0.0,
        'avg_fights_between_losses': 3.0,
        'cum_fights': 0.0,
    }
    for feat, default in defaults.items():
        df[f'R_{feat}'] = df[f'R_{feat}'].fillna(default)
        df[f'B_{feat}'] = df[f'B_{feat}'].fillna(default)

    # Compute diffs
    for feat in NEW_FEATURES:
        df[f'{feat}_dif'] = df[f'R_{feat}'] - df[f'B_{feat}']

    # Target
    df['target'] = (df['Winner'] == 'Red').astype(int)

    # Filter: both fighters must have >= 1 prior UFC fight
    df = df[(df['R_cum_fights'] >= 1) & (df['B_cum_fights'] >= 1)].copy()
    print(f'   Augmented dataset: {len(df):,} rows')

    print('\n[4] Correlation of new trajectory features with target (Red wins):')
    print(f'\n{"Feature":<35s}  {"Corr (R)":>9s}  {"Corr (B)":>9s}  {"Corr (diff)":>12s}')
    print('-' * 70)

    all_feat_corrs = []
    for feat in NEW_FEATURES:
        r_col   = f'R_{feat}'
        b_col   = f'B_{feat}'
        dif_col = f'{feat}_dif'
        r_corr  = round(float(df[r_col].corr(df['target'])),  4) if r_col in df.columns else None
        b_corr  = round(float(df[b_col].corr(df['target'])),  4) if b_col in df.columns else None
        d_corr  = round(float(df[dif_col].corr(df['target'])),4) if dif_col in df.columns else None
        print(f'{feat:<35s}  {str(r_corr):>9s}  {str(b_corr):>9s}  {str(d_corr):>12s}')
        all_feat_corrs.append({'feature': feat, 'R_corr': r_corr, 'B_corr': b_corr, 'diff_corr': d_corr})

    # Also check age_at_fight (R_age / B_age already there)
    age_r = round(float(df['R_age'].corr(df['target'])), 4)
    age_b = round(float(df['B_age'].corr(df['target'])), 4)
    age_d = df['R_age'] - df['B_age']
    age_dif_corr = round(float(age_d.corr(df['target'])), 4)
    print(f'{"age_at_fight (R/B from master)":<35s}  {age_r:>9.4f}  {age_b:>9.4f}  {age_dif_corr:>12.4f}')
    all_feat_corrs.append({'feature': 'age_at_fight', 'R_corr': age_r, 'B_corr': age_b, 'diff_corr': age_dif_corr})

    print()
    # Save augmented dataset — only keep key columns to avoid bloat
    save_cols = ['R_fighter', 'B_fighter', 'date', 'Winner', 'target', 'weight_class',
                 'R_age', 'B_age'] + \
                [f'R_{f}' for f in NEW_FEATURES] + \
                [f'B_{f}' for f in NEW_FEATURES] + \
                [f'{f}_dif' for f in NEW_FEATURES] + \
                ['R_cum_fights', 'B_cum_fights']
    save_cols = [c for c in save_cols if c in df.columns]

    out_path = os.path.join(OUT, 'augmented_dataset.csv')
    df[save_cols].to_csv(out_path, index=False)

    corr_path = os.path.join(OUT, 'trajectory_feature_correlations.csv')
    pd.DataFrame(all_feat_corrs).to_csv(corr_path, index=False)

    print(f'\n✓ Augmented dataset saved to {out_path}')
    print(f'✓ Correlation table saved to {corr_path}')
    print('=' * 60)
    print('  STEP 2 COMPLETE')
    print('=' * 60)

    return df, NEW_FEATURES

if __name__ == '__main__':
    main()
