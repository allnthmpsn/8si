"""
Temporal-leakage regression tests for training/train_model1.py.

Strategy: for a sampled (fighter, date) pair, recompute the same stat from a
copy of the source data truncated to information available strictly before
that date, and compare to the value the production pipeline actually
produced. A mismatch means the pipeline saw information from on/after the
fight's own date.

Per the remediation plan (8si_remediation_plan.md), both known leaks are now
fixed:
  - style stats (Phase 1): was a non-temporal snapshot merge with no date
    dimension; now an as-of merge via training/style_stats.py.
  - opp_quality (Phase 2): was scored by full-career (all-time) win rate;
    now scored by each opponent's career_win_rate as of the fight date.
"""
import math
import os

import numpy as np
import pandas as pd
import pytest

from training.train_model1 import DATA, compute_career_stats, compute_elo, compute_qa_stats
from training.style_stats import STYLE_STATS, compute_style_stats_asof

N_SAMPLES = 20
N_ELO_SAMPLES = 10
SEED = 42
TOL = 1e-6

# ─── Load source data once ──────────────────────────────────────────────────
career_df_full = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
career_df_full['date'] = pd.to_datetime(career_df_full['date'])
career_df_full = career_df_full.sort_values(['fighter', 'date']).reset_index(drop=True)

ufc_master_full = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
ufc_master_full['date'] = pd.to_datetime(ufc_master_full['date'])

career_stats_full = compute_career_stats(career_df_full)
elo_hist_full, _ = compute_elo(ufc_master_full)
qa_stats_full = compute_qa_stats(career_df_full, elo_hist_full)

# Sampling universe: fighters with enough history that pre-fight stats are
# non-default and comparisons are meaningful.
_with_cum = career_df_full.copy()
_with_cum['cum_fights'] = _with_cum.groupby('fighter').cumcount()
eligible_rows = _with_cum[_with_cum['cum_fights'] >= 3]
career_samples = eligible_rows.sample(n=N_SAMPLES, random_state=SEED).to_dict('records')

# opp_quality is most meaningfully checked with a fully-populated 5-opponent
# lookback window.
eligible_opp_rows = _with_cum[_with_cum['cum_fights'] >= 5]
opp_quality_samples = eligible_opp_rows.sample(n=10, random_state=SEED).to_dict('records')

eligible_elo_rows = ufc_master_full.dropna(subset=['R_fighter', 'B_fighter', 'date'])
elo_samples = eligible_elo_rows.sample(n=N_ELO_SAMPLES, random_state=SEED).to_dict('records')

gold_full = pd.read_csv(os.path.join(DATA, 'ufc_gold_dataset_final.csv'))
style_stats_full = compute_style_stats_asof(gold_full)
# Only rows with a real prior fight in this source are meaningful comparisons
# — NaN rows (debut / name not in ufc_gold_dataset_final.csv) are imputed
# downstream in train_model1.py, not something this test can check.
eligible_style_rows = style_stats_full.dropna(subset=['SLpM'])
style_samples = eligible_style_rows.sample(n=15, random_state=SEED).to_dict('records')


def _close_or_both_nan(a, b, tol=TOL):
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, abs_tol=tol)


def _career_stat_row(stats_df, fighter, date):
    rows = stats_df[(stats_df['fighter'] == fighter) & (stats_df['date'] == date)]
    assert len(rows) >= 1, f'no career-stat row for {fighter} on {date}'
    return rows.iloc[-1]


CAREER_STAT_COLS = [
    'cum_fights', 'career_win_rate', 'ko_finish_rate', 'sub_finish_rate',
    'career_finish_rate', 'last3_win_rate', 'last10_win_rate', 'last5_won',
    'last5_finish_rate', 'trend_score', 'layoff_days',
]


@pytest.mark.parametrize('sample', career_samples, ids=lambda s: f"{s['fighter']}@{s['date'].date()}")
def test_career_stats_no_leakage(sample):
    fighter, date = sample['fighter'], sample['date']

    # Truncate to this fighter's OWN rows up to and including the target
    # fight. None of CAREER_STAT_COLS (opp_quality excluded — see
    # test_opp_quality_no_leakage) depend on other fighters' rows, so only
    # this fighter's own future fights are a possible leak source here.
    trunc = career_df_full[(career_df_full['fighter'] == fighter) & (career_df_full['date'] <= date)]
    expected_stats = compute_career_stats(trunc)
    expected = _career_stat_row(expected_stats, fighter, date)
    actual = _career_stat_row(career_stats_full, fighter, date)

    for col in CAREER_STAT_COLS:
        assert math.isclose(float(expected[col]), float(actual[col]), abs_tol=TOL), (
            f'{col} leaked for {fighter} on {date.date()}: '
            f'expected(truncated)={expected[col]} actual(pipeline)={actual[col]}'
        )


@pytest.mark.parametrize('sample', career_samples, ids=lambda s: f"{s['fighter']}@{s['date'].date()}")
def test_qa_stats_no_leakage(sample):
    fighter, date = sample['fighter'], sample['date']

    trunc = career_df_full[(career_df_full['fighter'] == fighter) & (career_df_full['date'] <= date)]
    # elo_hist_full is reused as-is: opponents' elo_before at dates <= `date`
    # is unaffected by this fighter's own future fights.
    expected_stats = compute_qa_stats(trunc, elo_hist_full)
    expected = _career_stat_row(expected_stats, fighter, date)
    actual = _career_stat_row(qa_stats_full, fighter, date)

    for col in ['qa_win_rate', 'qa_finish_rate', 'qa_SLpM', 'qa_SApM']:
        assert math.isclose(float(expected[col]), float(actual[col]), abs_tol=TOL), (
            f'{col} leaked for {fighter} on {date.date()}: '
            f'expected(truncated)={expected[col]} actual(pipeline)={actual[col]}'
        )


@pytest.mark.parametrize(
    'sample', elo_samples,
    ids=lambda s: f"{s['R_fighter']}_v_{s['B_fighter']}@{s['date'].date()}",
)
def test_elo_no_leakage(sample):
    date = sample['date']
    trunc = ufc_master_full[ufc_master_full['date'] <= date]
    elo_hist_trunc, _ = compute_elo(trunc)

    for corner in ('R_fighter', 'B_fighter'):
        fighter = sample[corner]
        opponent = sample['B_fighter'] if corner == 'R_fighter' else sample['R_fighter']
        expected = elo_hist_trunc[
            (elo_hist_trunc['fighter'] == fighter)
            & (elo_hist_trunc['opponent'] == opponent)
            & (elo_hist_trunc['date'] == date)
        ]
        actual = elo_hist_full[
            (elo_hist_full['fighter'] == fighter)
            & (elo_hist_full['opponent'] == opponent)
            & (elo_hist_full['date'] == date)
        ]
        assert len(expected) >= 1 and len(actual) >= 1
        assert math.isclose(
            float(expected.iloc[-1]['elo_before']), float(actual.iloc[-1]['elo_before']), abs_tol=TOL
        ), f'elo_before leaked for {fighter} vs {opponent} on {date.date()}'


@pytest.mark.parametrize('sample', opp_quality_samples, ids=lambda s: f"{s['fighter']}@{s['date'].date()}")
def test_opp_quality_no_leakage(sample):
    """Phase 2 fix: each opponent is scored by THEIR career_win_rate as of
    the target fight's date, not their full-career (all-time) win rate."""
    fighter, date = sample['fighter'], sample['date']

    own_hist = career_df_full[career_df_full['fighter'] == fighter].sort_values('date').reset_index(drop=True)
    rank = own_hist.index[own_hist['date'] == date][-1]
    past_opps = own_hist.loc[max(0, rank - 5):rank - 1, 'opponent'].tolist()

    rates = []
    for opp in past_opps:
        opp_hist = career_df_full[(career_df_full['fighter'] == opp) & (career_df_full['date'] < date)]
        if len(opp_hist) > 0:
            rates.append(opp_hist['won'].sum() / len(opp_hist))
    expected = float(np.mean(rates)) if rates else 0.5

    actual = float(_career_stat_row(career_stats_full, fighter, date)['opp_quality'])
    assert math.isclose(expected, actual, abs_tol=TOL), (
        f'opp_quality leaked for {fighter} on {date.date()}: '
        f'expected(as-of)={expected} actual(pipeline)={actual}'
    )


@pytest.mark.parametrize('sample', style_samples, ids=lambda s: f"{s['fighter']}@{s['date'].date()}")
def test_style_stats_no_leakage(sample):
    fighter, date = sample['fighter'], sample['date']

    # Truncate the WIDE gold dataset to fights involving this fighter, up to
    # and including the target date. compute_style_stats_asof converts to
    # long format internally and computes per-fighter cumulative rates, so
    # this isolates whether the fighter's own future fights leaked in.
    trunc = gold_full[
        ((gold_full['Fighter_1'] == fighter) | (gold_full['Fighter_2'] == fighter))
        & (pd.to_datetime(gold_full['Event_Date']) <= date)
    ]
    expected_stats = compute_style_stats_asof(trunc)
    expected = _career_stat_row(expected_stats, fighter, date)
    actual = _career_stat_row(style_stats_full, fighter, date)

    for col in STYLE_STATS:
        assert _close_or_both_nan(float(expected[col]), float(actual[col])), (
            f'{col} leaked for {fighter} on {date.date()}: '
            f'expected(truncated)={expected[col]} actual(pipeline)={actual[col]}'
        )
