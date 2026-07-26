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


# ─── Observed-mask robustness (8SI v2 style_stats.py hardening) ────────────
# compute_style_stats_asof's raw source (ufc_gold_dataset_final.csv) has no
# producer script (docs/DATA_SOURCES.md) and currently has zero NaNs in its
# strike/TD columns, but a future re-scrape could easily introduce gaps.
# These tests inject a synthetic NaN and pin the exact expected behavior of
# the observed-mask design (see style_stats.py's docstring): a fight
# missing one raw column must (a) not corrupt ITS OWN "as of before this
# fight" aggregate, (b) be excluded entirely from later aggregates for the
# stat(s) that depend on the missing column — not silently zero-filled —
# and (c) leave every OTHER stat that doesn't depend on that column
# completely unaffected.
_NAN_TEST_FIGHTER = 'Aaron Riley'
_NAN_TEST_IDX = 500  # Aaron Riley vs Spencer Fisher, 2006-01-16 (Fighter_2, F2_Sig_Landed=9)


def test_style_stats_nan_does_not_poison_own_row():
    nan_injected = gold_full.copy()
    nan_injected.loc[_NAN_TEST_IDX, 'F2_Sig_Landed'] = np.nan

    clean_stats = compute_style_stats_asof(gold_full)
    nan_stats = compute_style_stats_asof(nan_injected)

    nan_date = pd.to_datetime(gold_full.loc[_NAN_TEST_IDX, 'Event_Date'])
    clean_row = _career_stat_row(clean_stats, _NAN_TEST_FIGHTER, nan_date)
    nan_row = _career_stat_row(nan_stats, _NAN_TEST_FIGHTER, nan_date)

    # This fight's own "as of before" aggregate depends only on Aaron
    # Riley's ONE earlier fight (2002-05-10) — NaN'ing THIS fight's own
    # sig_landed must not change it, and must never leave it NaN when real
    # prior data exists.
    for col in ('SLpM', 'Str_Acc'):
        assert not math.isnan(float(nan_row[col])), (
            f'{col} was unnecessarily NaN-poisoned at its own row ({nan_date.date()})'
        )
        assert math.isclose(float(clean_row[col]), float(nan_row[col]), abs_tol=TOL), (
            f'{col} at {nan_date.date()} changed after NaN-ing this fight\'s OWN value: '
            f'clean={clean_row[col]} nan_injected={nan_row[col]}'
        )


def test_style_stats_nan_excludes_not_zero_fills():
    nan_injected = gold_full.copy()
    nan_injected.loc[_NAN_TEST_IDX, 'F2_Sig_Landed'] = np.nan
    dropped = gold_full.drop(index=_NAN_TEST_IDX)

    clean_stats = compute_style_stats_asof(gold_full)
    nan_stats = compute_style_stats_asof(nan_injected)
    dropped_stats = compute_style_stats_asof(dropped)

    nan_date = pd.to_datetime(gold_full.loc[_NAN_TEST_IDX, 'Event_Date'])
    fighter_dates = sorted(pd.to_datetime(gold_full[
        (gold_full['Fighter_1'] == _NAN_TEST_FIGHTER) | (gold_full['Fighter_2'] == _NAN_TEST_FIGHTER)
    ]['Event_Date']))
    later_dates = [d for d in fighter_dates if d > nan_date]
    assert len(later_dates) >= 3, 'fixture assumption changed — need fights after the NaN-injected one'

    for date in later_dates:
        nan_row = _career_stat_row(nan_stats, _NAN_TEST_FIGHTER, date)
        dropped_row = _career_stat_row(dropped_stats, _NAN_TEST_FIGHTER, date)
        clean_row = _career_stat_row(clean_stats, _NAN_TEST_FIGHTER, date)

        # sig_landed-dependent stats: the NaN'd fight must be excluded
        # entirely from the running aggregate — same as if it had never
        # happened (dropped) — not silently treated as a zero contribution
        # (which would instead silently match a downward-biased number,
        # never exactly equal to either clean or dropped by coincidence).
        for col in ('SLpM', 'Str_Acc'):
            assert math.isclose(float(nan_row[col]), float(dropped_row[col]), abs_tol=TOL), (
                f'{col} for {_NAN_TEST_FIGHTER} on {date.date()}: NaN-injected={nan_row[col]} '
                f'expected(fight dropped entirely)={dropped_row[col]}'
            )
            assert not math.isclose(float(nan_row[col]), float(clean_row[col]), abs_tol=TOL), (
                f'{col} for {_NAN_TEST_FIGHTER} on {date.date()}: matched the unmodified value '
                f'exactly — the NaN-injected fight\'s real strikes should have changed this'
            )

        # Stats that don't depend on sig_landed at all must be completely
        # unaffected — this is the per-stat joint-mask isolation property:
        # a gap in one raw column shouldn't touch stats built from a
        # different column, even for the same fight.
        for col in ('SApM', 'TD_Avg', 'Sub_Avg', 'TD_Acc', 'Str_Def', 'TD_Def'):
            assert math.isclose(float(nan_row[col]), float(clean_row[col]), abs_tol=TOL), (
                f'{col} for {_NAN_TEST_FIGHTER} on {date.date()} unexpectedly changed: '
                f'clean={clean_row[col]} nan_injected={nan_row[col]}'
            )


def test_style_stats_snapshot_regression():
    """Pins known-correct values (captured from the parity-tested pipeline
    before the observed-mask refactor) for 5 real fighters on real,
    NaN-free data — catches any future change to compute_style_stats_asof
    that alters a real, already-correct value, even one that wouldn't be
    caught by the leakage/robustness tests above."""
    expected = [
        ('Tom Aspinall', '2025-10-25', {'SLpM': 8.0653951, 'SApM': 2.88828338, 'Str_Acc': 0.67272727,
                                         'Str_Def': 0.65359477, 'TD_Avg': 3.26975477, 'TD_Acc': 1.0,
                                         'TD_Def': 1.0, 'Sub_Avg': 1.63487738}),
        ('Chase Sherman', '2023-05-13', {'SLpM': 6.29587156, 'SApM': 6.85321101, 'Str_Acc': 0.46376077,
                                          'Str_Def': 0.51930502, 'TD_Avg': 0.10321101, 'TD_Acc': 0.5,
                                          'TD_Def': 0.66666667, 'Sub_Avg': 0.0}),
        ('Alexander Hernandez', '2025-09-13', {'SLpM': 4.35548275, 'SApM': 4.60120815, 'Str_Acc': 0.40958983,
                                                'Str_Def': 0.58296214, 'TD_Avg': 1.19791133, 'TD_Acc': 0.36111111,
                                                'TD_Def': 0.73170732, 'Sub_Avg': 0.09214703}),
        ('Jack Shore', '2024-11-02', {'SLpM': 3.74591652, 'SApM': 2.31941924, 'Str_Acc': 0.58703072,
                                       'Str_Def': 0.56619145, 'TD_Avg': 3.10344828, 'TD_Acc': 0.3877551,
                                       'TD_Def': 0.76, 'Sub_Avg': 0.65335753}),
        ('Gilbert Burns', '2025-05-17', {'SLpM': 3.17227249, 'SApM': 3.57179443, 'Str_Acc': 0.48511749,
                                          'Str_Def': 0.52755194, 'TD_Avg': 2.10005122, 'TD_Acc': 0.37614679,
                                          'TD_Def': 0.53658537, 'Sub_Avg': 0.46098685}),
    ]

    for fighter, date, vals in expected:
        row = _career_stat_row(style_stats_full, fighter, pd.Timestamp(date))
        for col, expected_val in vals.items():
            actual_val = float(row[col])
            assert math.isclose(actual_val, expected_val, abs_tol=1e-6), (
                f'{col} for {fighter} on {date}: expected {expected_val}, got {actual_val} '
                f'— compute_style_stats_asof\'s output changed for real, already-correct data'
            )


# ─── Round-data leakage harness (8SI v2 Stage 1.3) ──────────────────────────
# data/round_stats.parquet (training/ingest_rounds.py) is the source for
# every Stage 2 feature family. No Stage 2 feature function exists yet, so
# rather than a per-feature test, this is the REUSABLE harness those
# functions plug into as they land — per the v2 spec's Stage 1.3
# requirement to "write the test harness now, parameterized to accept the
# Stage 2 feature functions as they land."
#
# Contract a round-derived feature function must satisfy:
#   feature_fn(round_stats_df) -> DataFrame[fighter, date, <feature cols>],
#   one row per (fighter, fight), value computed using ONLY that fighter's
#   ROUND rows strictly before `date` (shift(1) discipline, same as
#   compute_career_stats()/compute_style_stats_asof() — see this file's
#   module docstring for the general strategy).
round_stats_full = pd.read_parquet(os.path.join(DATA, 'round_stats.parquet'))


def assert_round_feature_no_leakage(feature_fn, fighter, date, feature_cols, tol=TOL):
    """For `feature_fn` satisfying the contract above, assert its output at
    (fighter, date) is unchanged whether computed from the full
    round_stats_full or from a copy truncated to this fighter's own rounds
    up to and including `date` — a mismatch means a round from on/after
    `date` leaked into the 'as of' value. Mirrors
    test_style_stats_no_leakage's truncate-and-recompute strategy exactly."""
    date = pd.Timestamp(date)
    trunc = round_stats_full[(round_stats_full['fighter'] == fighter) & (round_stats_full['date'] <= date)]
    expected_stats = feature_fn(trunc)
    expected = _career_stat_row(expected_stats, fighter, date)
    actual = _career_stat_row(feature_fn(round_stats_full), fighter, date)
    for col in feature_cols:
        assert _close_or_both_nan(float(expected[col]), float(actual[col])), (
            f'{col} leaked for {fighter} on {date.date()}: '
            f'expected(truncated)={expected[col]} actual(full)={actual[col]}'
        )


def _toy_kd_cumulative(round_stats):
    """Proof-of-concept round-derived feature satisfying the contract
    above: cumulative knockdowns landed per fighter, as-of each fight
    (round-level rows summed to fight-level, then shift(1) across fights —
    the same two-step pattern any Stage 2.1 'per15' family will need).
    Exists purely to exercise assert_round_feature_no_leakage() with a real
    function; Stage 2's actual feature functions replace this and are
    tested the same way."""
    fight_level = round_stats.groupby(['fighter', 'date'], as_index=False)['kd'].sum()
    fight_level = fight_level.sort_values(['fighter', 'date']).reset_index(drop=True)
    g = fight_level.groupby('fighter', sort=False)
    fight_level['kd_cum'] = g['kd'].cumsum() - fight_level['kd']
    return fight_level[['fighter', 'date', 'kd_cum']]


_round_stats_eligible = round_stats_full[round_stats_full['date'] >= '2015-01-01']
_round_feature_samples = (
    _round_stats_eligible[['fighter', 'date']]
    .drop_duplicates()
    .groupby('fighter').filter(lambda g: len(g) >= 3)
    .sample(n=15, random_state=SEED)
    .to_dict('records')
)


@pytest.mark.parametrize(
    'sample', _round_feature_samples,
    ids=lambda s: f"{s['fighter']}@{pd.Timestamp(s['date']).date()}",
)
def test_round_derived_feature_no_leakage(sample):
    assert_round_feature_no_leakage(_toy_kd_cumulative, sample['fighter'], sample['date'], ['kd_cum'])


# ─── KD/damage family (8SI v2 Stage 2.1) ────────────────────────────────────
# training/features_kd.py doesn't fit assert_round_feature_no_leakage's
# contract directly: each fighter's feature value depends on the OPPONENT's
# stats too (from the same shared past fights, via a self-join), plus an
# external ufc_fight_results.csv merge for fight duration and a name_map.csv
# canonical-name join — none of which the generic single-fighter-column
# harness above accounts for. Same truncate-and-recompute strategy as
# test_style_stats_no_leakage, adapted: truncate round_stats.parquet to
# FIGHTS INVOLVING the sampled fighter (not just their own 'fighter'-column
# rows) up to and including the target date, so each of their own past
# fights keeps BOTH corners for the self-join to work correctly on the
# truncated data — recompute via the exact same compute_kd_features_asof()
# (pointed at a temp copy), compare to the full-pipeline value.
from training.features_kd import compute_kd_features_asof, KD_FEATURES, _load_name_map  # noqa: E402

_kd_name_map = _load_name_map()
_canonical_to_raw = {}
for _raw, _canon in _kd_name_map.items():
    _canonical_to_raw.setdefault(_canon, set()).add(_raw)


def _truncated_round_stats_path(fighter, date, tmp_path):
    """Write round_stats.parquet truncated to FIGHTS INVOLVING `fighter`
    (both corners kept, so any same-fight self-join still works) up to
    and including `date`, to a temp file — reusable across every
    round-derived feature family with an opponent dependency (KD,
    grappling, ...), since they all share this exact truncation need."""
    raw_names = _canonical_to_raw.get(fighter, {fighter})
    own_events = round_stats_full[
        round_stats_full['fighter'].isin(raw_names) & (round_stats_full['date'] <= date)
    ][['event', 'bout']].drop_duplicates()
    trunc = round_stats_full.merge(own_events, on=['event', 'bout']).query('date <= @date')
    trunc_path = tmp_path / 'round_stats_trunc.parquet'
    trunc.to_parquet(trunc_path, index=False)
    return str(trunc_path)


_kd_full = compute_kd_features_asof()
_kd_eligible = _kd_full.dropna(subset=['damage_ratio']).groupby('fighter').size()
_kd_eligible = _kd_eligible[_kd_eligible >= 5].index.tolist()
_kd_samples = [
    {'fighter': f, 'date': _kd_full[_kd_full['fighter'] == f].sample(n=1, random_state=SEED).iloc[0]['date']}
    for f in pd.Series(_kd_eligible).sample(n=8, random_state=SEED).tolist()
]


@pytest.mark.parametrize(
    'sample', _kd_samples,
    ids=lambda s: f"{s['fighter']}@{pd.Timestamp(s['date']).date()}",
)
def test_kd_features_no_leakage(sample, tmp_path):
    fighter, date = sample['fighter'], pd.Timestamp(sample['date'])
    trunc_path = _truncated_round_stats_path(fighter, date, tmp_path)

    expected_stats = compute_kd_features_asof(round_stats_path=trunc_path)
    expected = _career_stat_row(expected_stats, fighter, date)
    actual = _career_stat_row(_kd_full, fighter, date)

    for col in KD_FEATURES:
        assert _close_or_both_nan(float(expected[col]), float(actual[col])), (
            f'{col} leaked for {fighter} on {date.date()}: '
            f'expected(truncated)={expected[col]} actual(full)={actual[col]}'
        )


# ─── Control & grappling exposure family (8SI v2 Stage 2.2) ────────────────
from training.features_grappling import compute_grappling_features_asof, GRAPPLING_FEATURES  # noqa: E402

_gr_full = compute_grappling_features_asof()
_gr_eligible = _gr_full.dropna(subset=['ctrl_pct_for']).groupby('fighter').size()
_gr_eligible = _gr_eligible[_gr_eligible >= 5].index.tolist()
_gr_samples = [
    {'fighter': f, 'date': _gr_full[_gr_full['fighter'] == f].sample(n=1, random_state=SEED).iloc[0]['date']}
    for f in pd.Series(_gr_eligible).sample(n=8, random_state=SEED).tolist()
]


@pytest.mark.parametrize(
    'sample', _gr_samples,
    ids=lambda s: f"{s['fighter']}@{pd.Timestamp(s['date']).date()}",
)
def test_grappling_features_no_leakage(sample, tmp_path):
    fighter, date = sample['fighter'], pd.Timestamp(sample['date'])
    trunc_path = _truncated_round_stats_path(fighter, date, tmp_path)

    expected_stats = compute_grappling_features_asof(round_stats_path=trunc_path)
    expected = _career_stat_row(expected_stats, fighter, date)
    actual = _career_stat_row(_gr_full, fighter, date)

    for col in GRAPPLING_FEATURES:
        assert _close_or_both_nan(float(expected[col]), float(actual[col])), (
            f'{col} leaked for {fighter} on {date.date()}: '
            f'expected(truncated)={expected[col]} actual(full)={actual[col]}'
        )
