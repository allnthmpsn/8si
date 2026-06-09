#!/usr/bin/env python3
"""
Step 3 — Career Archetype Clustering
Clusters fighters into archetypes using K-means on trajectory features.
Saves: experiments/research/archetypes/ (plots + CSV)
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import gc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
OUT  = os.path.join(ROOT, 'experiments', 'research', 'archetypes')

def compute_trajectory_features(career_df):
    """Same as step2 — returns per-fight trajectory features per fighter."""
    df = career_df.sort_values(['fighter', 'date']).copy().reset_index(drop=True)
    df['_ko']  = ((df['won'] == 1) & df['method'].str.contains('KO|TKO', case=False, na=False)).astype(float)
    df['_sub'] = ((df['won'] == 1) & df['method'].str.contains('Sub|Submission', case=False, na=False)).astype(float)
    df['_fin'] = ((df['won'] == 1) & df['method'].str.contains('KO|TKO|Sub', case=False, na=False)).astype(float)
    g = df.groupby('fighter', sort=False)
    df['cum_fights'] = g.cumcount()
    df['_cs_fin'] = g['_fin'].cumsum() - df['_fin']
    df['_cs_won'] = g['won'].cumsum()  - df['won']
    safe_n = df['cum_fights'].clip(lower=1)
    df['career_finish_rate']  = np.where(df['cum_fights'] > 0, df['_cs_fin'] / safe_n, 0.0)
    df['career_win_rate_traj'] = np.where(df['cum_fights'] > 0, df['_cs_won'] / safe_n, 0.5)
    def _roll(series, w, default):
        return series.shift(1).rolling(w, min_periods=1).mean().fillna(default)
    df['last5_won_traj']         = g['won'].transform(lambda x: _roll(x, 5, 0.5))
    df['last5_finish_rate_traj'] = g['_fin'].transform(lambda x: _roll(x, 5, 0.0))
    df['finish_rate_trend']      = df['last5_finish_rate_traj'] - df['career_finish_rate']
    df['win_rate_l5_vs_career']  = df['last5_won_traj'] - df['career_win_rate_traj']

    def _fights_since_finish(series_fin):
        shifted = series_fin.shift(1)
        result  = []
        count   = 0
        for v in shifted:
            if pd.isna(v):
                result.append(0.0)
            else:
                result.append(float(count))
                if v == 1:
                    count = 0
                else:
                    count += 1
        return pd.Series(result, index=series_fin.index)

    df['fights_since_finish'] = g['_fin'].transform(_fights_since_finish)

    def _max_losing_streak(series_won):
        shifted = series_won.shift(1)
        cur_streak = 0
        max_streak = 0
        result = []
        for v in shifted:
            if pd.isna(v):
                result.append(max_streak)
            else:
                if v == 0:
                    cur_streak += 1
                    max_streak = max(max_streak, cur_streak)
                else:
                    cur_streak = 0
                result.append(max_streak)
        return pd.Series(result, index=series_won.index)

    df['longest_lose_streak_ever'] = g['won'].transform(_max_losing_streak)
    df.drop(columns=['_ko','_sub','_fin','_cs_won','_cs_fin'], inplace=True)
    return df


def get_current_fighter_profile(df_traj):
    """Get the most recent trajectory snapshot per fighter (their current profile)."""
    latest = df_traj.sort_values(['fighter', 'date']).groupby('fighter').last().reset_index()
    return latest


def main():
    print('=' * 60)
    print('  STEP 3 — Career Archetype Clustering')
    print('=' * 60)

    print('\n[1] Loading and computing trajectory features...')
    career_df = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_df['date'] = pd.to_datetime(career_df['date'])
    career_df = career_df.sort_values(['fighter', 'date']).reset_index(drop=True)
    print(f'   Career rows: {len(career_df):,}  |  Fighters: {career_df["fighter"].nunique():,}')

    df_traj = compute_trajectory_features(career_df)

    # Get current profile per fighter (latest snapshot)
    current = get_current_fighter_profile(df_traj)
    print(f'   Fighter profiles: {len(current):,}')

    # Merge age from DOB in ufc_fighters_final_updated.csv
    from datetime import datetime
    style_df = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final_updated.csv'))
    style_df = style_df.drop_duplicates(subset=['Fighter_Name'], keep='last')
    style_df['DOB'] = pd.to_datetime(style_df['DOB'], errors='coerce')
    today = datetime.now()
    style_df['current_age'] = ((today - style_df['DOB']).dt.days / 365.25).round(1)
    current = current.merge(
        style_df[['Fighter_Name', 'current_age']].rename(columns={'Fighter_Name': 'fighter'}),
        on='fighter', how='left'
    )
    current['current_age'] = pd.to_numeric(current['current_age'], errors='coerce').fillna(29.0)

    # Clustering features
    CLUSTER_FEATS = [
        'current_age',
        'career_win_rate_traj',
        'career_finish_rate',
        'finish_rate_trend',
        'win_rate_l5_vs_career',
        'fights_since_finish',
    ]
    for f in CLUSTER_FEATS:
        if f not in current.columns:
            current[f] = 0.0
        current[f] = pd.to_numeric(current[f], errors='coerce').fillna(0.0)

    X_cluster = current[CLUSTER_FEATS].values

    print('\n[2] Scaling and trying K = 4, 5, 6...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    inertias = {}
    for k in [4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_
        print(f'   K={k}  inertia={km.inertia_:.1f}')

    # Choose K=5 as good balance
    K = 5
    print(f'\n[3] Fitting final KMeans K={K}...')
    km_final = KMeans(n_clusters=K, random_state=42, n_init=20)
    km_final.fit(X_scaled)
    current['cluster'] = km_final.labels_

    print('\n[4] Cluster summary (mean stats):')
    cluster_stats = current.groupby('cluster')[CLUSTER_FEATS].mean().round(3)
    cluster_sizes  = current.groupby('cluster').size().rename('n_fighters')
    cluster_summary = cluster_stats.join(cluster_sizes)
    print(cluster_summary.to_string())

    # Label clusters by ranking on key dimensions
    # Sort clusters by career_win_rate_traj to assign ordered labels
    LABELS = {}
    rank_by = cluster_summary[['career_win_rate_traj', 'career_finish_rate',
                                 'win_rate_l5_vs_career', 'current_age']].copy()

    for c in range(K):
        cdata = cluster_summary.loc[c]
        age   = cdata.get('current_age', 29)
        wr    = cdata.get('career_win_rate_traj', 0.5)
        fr    = cdata.get('career_finish_rate', 0.3)
        trend = cdata.get('win_rate_l5_vs_career', 0.0)
        fsf   = cdata.get('fights_since_finish', 2.0)

        # Assign label by distinctive feature
        if wr < 0.50:
            label = 'Journeyman'
        elif wr >= 0.80 and fr >= 0.60:
            label = 'Elite Finisher'
        elif wr >= 0.65 and age >= 33:
            label = 'Veteran Contender'
        elif wr >= 0.65 and trend <= -0.25:
            label = 'Fading Contender'
        elif wr >= 0.65 and fsf >= 5:
            label = 'Decision Specialist'
        else:
            label = 'Active Performer'

        LABELS[c] = label
        print(f'   Cluster {c} ({int(cluster_sizes[c])} fighters) → "{label}" | age={age:.1f}  wr={wr:.3f}  fr={fr:.3f}  trend={trend:+.3f}  fsf={fsf:.1f}')

    current['archetype'] = current['cluster'].map(LABELS)

    # ── Win rate by archetype ──────────────────────────────────────────────────
    print('\n[5] Win rate by archetype (actual career win rate):')
    for arch, grp in current.groupby('archetype'):
        n    = len(grp)
        wr   = grp['career_win_rate_traj'].mean()
        fr   = grp['career_finish_rate'].mean()
        age  = grp['current_age'].mean()
        print(f'   {arch:<25s}  n={n:4d}  avg_wr={wr:.3f}  avg_fr={fr:.3f}  avg_age={age:.1f}')

    # ── Plot: avg career win rate over age per archetype ─────────────────────
    print('\n[6] Generating archetype plots...')

    # Merge archetype back onto the full trajectory
    archetype_map = current[['fighter', 'archetype', 'cluster']].copy()
    df_merged = df_traj.merge(archetype_map, on='fighter', how='left')
    df_merged = df_merged.dropna(subset=['archetype'])

    # Bin age to decades for rolling win rate over age
    df_merged['age_bin'] = pd.to_numeric(
        df_merged.get('current_age', pd.Series([29]*len(df_merged))),
        errors='coerce'
    ).fillna(29.0).round(0)

    # For each archetype, compute average rolling win rate over cum_fights (proxy for age)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('UFC Fighter Career Archetypes', fontsize=13, fontweight='bold')

    # Plot 1: career_win_rate_traj over cum_fights bins (0-5, 5-10, 10-15, 15-20, 20+)
    ax = axes[0]
    fight_bins = [0, 3, 6, 10, 15, 20, 999]
    bin_labels = ['1-3', '4-6', '7-10', '11-15', '16-20', '20+']
    df_merged['fight_bin'] = pd.cut(df_merged['cum_fights'], bins=fight_bins, labels=bin_labels, right=True)

    archetype_order = sorted(df_merged['archetype'].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(archetype_order)))
    for i, arch in enumerate(archetype_order):
        grp = df_merged[df_merged['archetype'] == arch]
        win_by_bin = grp.groupby('fight_bin', observed=True)['career_win_rate_traj'].mean()
        win_by_bin.plot(ax=ax, marker='o', label=arch, color=colors[i], linewidth=2)

    ax.set_xlabel('Career fight bucket (pre-fight cumulative)', fontsize=10)
    ax.set_ylabel('Avg Career Win Rate (pre-fight)', fontsize=10)
    ax.set_title('Win Rate Trajectory by Archetype', fontsize=11)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.grid(alpha=0.2)

    # Plot 2: finish rate over career
    ax2 = axes[1]
    for i, arch in enumerate(archetype_order):
        grp = df_merged[df_merged['archetype'] == arch]
        fin_by_bin = grp.groupby('fight_bin', observed=True)['career_finish_rate'].mean()
        fin_by_bin.plot(ax=ax2, marker='s', label=arch, color=colors[i], linewidth=2)

    ax2.set_xlabel('Career fight bucket (pre-fight cumulative)', fontsize=10)
    ax2.set_ylabel('Avg Career Finish Rate (pre-fight)', fontsize=10)
    ax2.set_title('Finish Rate Trajectory by Archetype', fontsize=11)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plot1_path = os.path.join(OUT, 'archetype_trajectories.png')
    plt.savefig(plot1_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'   Saved: {plot1_path}')

    # Plot 2: Cluster scatter
    fig2, ax3 = plt.subplots(figsize=(9, 6))
    for i, arch in enumerate(archetype_order):
        grp = current[current['archetype'] == arch]
        ax3.scatter(grp['current_age'], grp['career_win_rate_traj'],
                    label=arch, alpha=0.5, s=20, color=colors[i])
    ax3.set_xlabel('Current Age', fontsize=11)
    ax3.set_ylabel('Career Win Rate', fontsize=11)
    ax3.set_title('Fighter Archetypes: Age vs Career Win Rate', fontsize=12, fontweight='bold')
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.2)
    plot2_path = os.path.join(OUT, 'archetype_scatter.png')
    plt.savefig(plot2_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'   Saved: {plot2_path}')

    # Save fighter archetype assignments
    fighter_archetypes_path = os.path.join(OUT, 'fighter_archetypes.csv')
    save_cols = ['fighter', 'cluster', 'archetype', 'current_age', 'cum_fights',
                 'career_win_rate_traj', 'career_finish_rate', 'win_rate_l5_vs_career',
                 'finish_rate_trend', 'fights_since_finish', 'longest_lose_streak_ever']
    save_cols = [c for c in save_cols if c in current.columns]
    current[save_cols].to_csv(fighter_archetypes_path, index=False)
    print(f'   Saved: {fighter_archetypes_path}')

    # Save cluster summary
    cluster_summary_path = os.path.join(OUT, 'cluster_summary.csv')
    cluster_summary['archetype'] = cluster_summary.index.map(LABELS)
    cluster_summary.to_csv(cluster_summary_path)
    print(f'   Saved: {cluster_summary_path}')

    print('\n' + '=' * 60)
    print('  STEP 3 COMPLETE')
    print('=' * 60)

    gc.collect()
    return current, LABELS

if __name__ == '__main__':
    main()
