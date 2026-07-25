"""
features/build.py — shared feature-computation pieces (8SI Phase 5 Part B).

get_fighter_state_asof() is the actual new capability: a point-in-time
query, "what would this fighter's career/QA/style/Elo/got_finished_rate
feature vector have been as of any given date." Built on top of
training/train_model1.py's own compute_career_stats()/compute_qa_stats()/
compute_got_finished_rate() and training/style_stats.compute_style_stats_asof()
— not reimplemented — using a synthetic as-of-date row per source table
(same technique as backend/main.py's Phase 5 Part A QA-stats fix). This is
what makes train and serve "unified": the SAME functions produce a training
row's features and a live (or historical, for the parity test) as-of-date
query.

Shared constants/small formulas (WC_ORDER, layoff buckets, etc.) live in
features/constants.py, not here — this module imports compute_*() FROM
train_model1.py, so train_model1.py can't also import constants back from
here without a circular import; constants.py has no dependency on either
side, so everyone can safely import from it.

NOT done here: get_fighter_state_asof() is not wired into backend/main.py's
live /predict or bet_recommendation() paths in this pass — see
docs/DECISIONS.md. It recomputes from a full synthetic-row pass per call,
which is fine for a test/analysis script and not yet optimized for a live
per-request path.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from training.train_model1 import (
    compute_career_stats, compute_qa_stats, compute_got_finished_rate, compute_elo,
    ELO_K, ELO_METHOD_MULTIPLIERS, ELO_LAYOFF_REGRESSION,
)
from training.style_stats import compute_style_stats_asof, STYLE_STATS


# ─────────────────────────────────────────────────────────────────────────────
# Point-in-time fighter state
# ─────────────────────────────────────────────────────────────────────────────
class DataBundle:
    """Loads career_fights_updated.csv, ufc-master.csv (-> Elo), and
    ufc_gold_dataset_final.csv once; passed into get_fighter_state_asof()
    so repeated calls don't reload CSVs from disk."""

    def __init__(self, data_dir):
        self.career_df = pd.read_csv(os.path.join(data_dir, 'career_fights_updated.csv'))
        self.career_df['date'] = pd.to_datetime(self.career_df['date'])
        self.career_df = self.career_df.sort_values(['fighter', 'date']).reset_index(drop=True)

        master = pd.read_csv(os.path.join(data_dir, 'ufc-master.csv'), low_memory=False)
        master['date'] = pd.to_datetime(master['date'])
        # Must match build_dataset()'s own (promoted, 8SI Phase 7) Elo config
        # exactly, or tests/test_train_serve_parity.py would compare two
        # different Elo formulas against each other.
        self.elo_hist_df, self.elo_curr_df = compute_elo(
            master, K=ELO_K, base=1500.0,
            method_multipliers=ELO_METHOD_MULTIPLIERS, layoff_regression=ELO_LAYOFF_REGRESSION,
        )

        self.gold_df = pd.read_csv(os.path.join(data_dir, 'ufc_gold_dataset_final.csv'))


def _synthetic_row(fighter_name, as_of_date):
    return pd.DataFrame([{
        'fighter': fighter_name, 'opponent': '__asof__', 'date': as_of_date,
        'won': 0, 'method': '', 'got_finish': 0,
    }])


def _elo_asof(fighter_name, as_of_date, elo_hist_df,
               base=1500.0, layoff_regression=ELO_LAYOFF_REGRESSION):
    """
    elo_after of the fighter's last fight strictly before as_of_date is
    normally identical to elo_before of their NEXT real fight (see
    compute_elo()) — EXCEPT when layoff_regression is active and the gap
    since that last fight exceeds its threshold, in which case
    compute_elo() regresses elo_before toward `base` for that next fight.
    Must replicate that same regression here, or an as-of-today (or
    as-of-any-date) query would silently disagree with what build_dataset()
    actually merges onto a real fight after a long layoff — this is
    exactly the mismatch tests/test_train_serve_parity.py caught for a
    real fighter (Jon Jones's 2023 comeback) before this fix.
    """
    hist = elo_hist_df[(elo_hist_df['fighter'] == fighter_name) & (elo_hist_df['date'] < as_of_date)]
    if len(hist) == 0:
        return {'elo': base, 'elo_trend': 0.0}
    hist = hist.sort_values('date')
    cur = float(hist.iloc[-1]['elo_after'])
    last_date = hist.iloc[-1]['date']

    if layoff_regression is not None:
        days_threshold, regress_pct = layoff_regression
        if (as_of_date - last_date).days > days_threshold:
            cur = cur + regress_pct * (base - cur)

    trend = float(cur - hist.iloc[-3]['elo_before']) if len(hist) >= 3 else 0.0
    return {'elo': cur, 'elo_trend': trend}


def get_fighter_state_asof(fighter_name, as_of_date, bundle: DataBundle):
    """
    fighter_name's full feature vector (career stats, QA stats, style
    stats, Elo, got_finished_rate) as of as_of_date, using ONLY
    information strictly before as_of_date — the same point-in-time
    discipline training/train_model1.py's build_dataset() uses for a
    training row, via the SAME underlying functions.

    Returns a flat dict; style-stat keys are NaN if the fighter has no
    prior fight in ufc_gold_dataset_final.csv, got_finished_rate is NaN if
    they have no prior loss yet — callers must impute these themselves
    (see train_model1._impute_by_weight_class for the trainer's approach),
    not fillna(0) blindly.
    """
    as_of_date = pd.Timestamp(as_of_date)
    synth = _synthetic_row(fighter_name, as_of_date)

    # Career stats + opp_quality need the FULL truncated career_df (other
    # fighters' rows are how opp_quality's opponent-lookback resolves),
    # not just this fighter's own history.
    truncated = bundle.career_df[bundle.career_df['date'] < as_of_date]
    career_aug = pd.concat([truncated, synth], ignore_index=True)
    career_stats = compute_career_stats(career_aug)
    career_row = career_stats[
        (career_stats['fighter'] == fighter_name) & (career_stats['date'] == as_of_date)
    ].iloc[-1]

    gf = compute_got_finished_rate(career_aug)
    gf_row = gf[(gf['fighter'] == fighter_name) & (gf['date'] == as_of_date)].iloc[-1]

    elo_hist_truncated = bundle.elo_hist_df[bundle.elo_hist_df['date'] < as_of_date]
    qa_stats = compute_qa_stats(career_aug, elo_hist_truncated)
    qa_row = qa_stats[(qa_stats['fighter'] == fighter_name) & (qa_stats['date'] == as_of_date)].iloc[-1]

    # <=, not <: the trainer's own merge_asof(direction='backward') includes
    # an exact-date match — i.e. the fighter's OWN gold-dataset row for this
    # exact fight, if present, whose pre-fight cumulative value already
    # correctly excludes that fight's own contribution (compute_style_stats_asof's
    # shift(1) discipline). Truncating strictly-before here would silently
    # drop that row and fall back to one fight earlier, undercounting.
    style_truncated = bundle.gold_df[pd.to_datetime(bundle.gold_df['Event_Date']) <= as_of_date]
    style_stats = compute_style_stats_asof(style_truncated)
    style_rows = style_stats[style_stats['fighter'] == fighter_name]
    if len(style_rows) > 0:
        style_row = style_rows.sort_values('date').iloc[-1]
        style_vals = {s: float(style_row[s]) if pd.notna(style_row[s]) else float('nan') for s in STYLE_STATS}
    else:
        style_vals = {s: float('nan') for s in STYLE_STATS}

    elo_vals = _elo_asof(fighter_name, as_of_date, bundle.elo_hist_df)

    return {
        'cum_fights':          float(career_row['cum_fights']),
        'career_win_rate':     float(career_row['career_win_rate']),
        'ko_finish_rate':      float(career_row['ko_finish_rate']),
        'sub_finish_rate':     float(career_row['sub_finish_rate']),
        'last3_win_rate':      float(career_row['last3_win_rate']),
        'last10_win_rate':     float(career_row['last10_win_rate']),
        'last5_won':           float(career_row['last5_won']),
        'last5_finish_rate':   float(career_row['last5_finish_rate']),
        'trend_score':         float(career_row['trend_score']),
        'opp_quality':         float(career_row['opp_quality']),
        'layoff_days':         float(career_row['layoff_days']),
        'qa_win_rate':         float(qa_row['qa_win_rate']),
        'qa_finish_rate':      float(qa_row['qa_finish_rate']),
        'qa_SLpM':             float(qa_row['qa_SLpM']),
        'qa_SApM':             float(qa_row['qa_SApM']),
        'got_finished_rate':   float(gf_row['got_finished_rate']) if pd.notna(gf_row['got_finished_rate']) else float('nan'),
        'elo':                 elo_vals['elo'],
        'elo_trend':           elo_vals['elo_trend'],
        **style_vals,
    }
