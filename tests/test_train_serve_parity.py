"""
tests/test_train_serve_parity.py — 8SI Phase 5 Part B acceptance test.

For sampled historical fights, assert that features/build.get_fighter_state_asof()
(the new point-in-time query function backend/main.py would eventually use)
produces the same per-fighter feature values as train_model1.build_dataset()
(the trainer's existing bulk-training path) — both ultimately call the same
compute_career_stats()/compute_qa_stats()/compute_got_finished_rate()/
compute_style_stats_asof() functions, so this is really testing that the
as-of-date synthetic-row wrapper in features/build.py reproduces what the
trainer's own merge_asof-based pipeline computes for a real training row.

Samples are restricted to fights where get_fighter_state_asof() itself
returns zero NaN across all 8 style-stat columns for BOTH fighters (checked
directly, not inferred from R/B_style_missing — that flag is only set from
R_SLpM specifically, but _impute_by_weight_class() weight-class-median-fills
every individual NaN cell across all 8 columns regardless of that flag, so
a fighter can have style_missing == 0 and still have e.g. an individually-NaN
TD_Def from a real gold-dataset row with zero recorded opponent attempts so
far). This test isn't re-validating Phase 1's imputation (covered by
tests/test_no_leakage.py) — it's purely about point-in-time correctness, so
samples are picked to avoid the imputation path entirely rather than
reproduce it here.
"""
import math
import os

import pandas as pd
import pytest

from training.train_model1 import DATA, TRAIN_START, TRAIN_CUTOFF, build_dataset
from features.build import DataBundle, get_fighter_state_asof, STYLE_STATS

N_SAMPLES = 10
SEED = 42
TOL = 1e-6

df, _, _ = build_dataset(TRAIN_START, TRAIN_CUTOFF, DATA)
bundle = DataBundle(DATA)

# career_fights_updated.csv and ufc-master.csv aren't perfectly 1:1 synced —
# a fighter's UFC fight can be missing its own row in career_df (data gap,
# not a bug). When that happens, the trainer's merge_asof(direction=
# 'backward') anchors on the fighter's last ACTUAL career_df row (whatever
# that row's own date is), while get_fighter_state_asof() counts everything
# strictly before the target date — different, defensible semantics, not
# comparable. Restrict samples to fights where both fighters' latest
# career_df row IS dated exactly on the target fight's date, so the two
# approaches are answering the same question.
#
# Also exclude (fighter, date) rows that appear more than once in career_df
# for the target date — a documented, real data-quality quirk (see
# docs/DATA_SOURCES.md's "~2,235 rows share a (fighter, date) combination"
# gotcha, e.g. "Zachary Reese" vs. "Zach Reese" logged as two separate
# opponent entries for the same fight): merge_asof(backward) breaks the tie
# by picking whichever duplicate sorts last, which get_fighter_state_asof()
# doesn't try to replicate.
_last_career_date = bundle.career_df.groupby('fighter')['date'].max().to_dict()
_date_counts = bundle.career_df.groupby(['fighter', 'date']).size().to_dict()


def _career_df_in_sync(row):
    d = pd.Timestamp(row['date'])
    r, b = row['R_fighter'], row['B_fighter']
    return (
        _last_career_date.get(r) == d and _last_career_date.get(b) == d
        and _date_counts.get((r, d), 0) == 1
        and _date_counts.get((b, d), 0) == 1
    )


_candidates = df[
    (df['R_cum_fights'] >= 3) & (df['B_cum_fights'] >= 3)
    & (df['R_style_missing'] == 0) & (df['B_style_missing'] == 0)
    & (df['R_gf_missing'] == 0) & (df['B_gf_missing'] == 0)
].sample(frac=1, random_state=SEED).to_dict('records')

samples = []
_states_by_sample = {}
for _row in _candidates:
    if len(samples) >= N_SAMPLES:
        break
    if not _career_df_in_sync(_row):
        continue
    _r_state = get_fighter_state_asof(_row['R_fighter'], pd.Timestamp(_row['date']), bundle)
    _b_state = get_fighter_state_asof(_row['B_fighter'], pd.Timestamp(_row['date']), bundle)
    if any(math.isnan(_r_state[s]) for s in STYLE_STATS) or any(math.isnan(_b_state[s]) for s in STYLE_STATS):
        continue
    samples.append(_row)
    _key = f"{_row['R_fighter']}_v_{_row['B_fighter']}@{pd.Timestamp(_row['date']).date()}"
    _states_by_sample[_key] = (_r_state, _b_state)

CAREER_COLS = [
    'cum_fights', 'career_win_rate', 'ko_finish_rate', 'sub_finish_rate',
    'last3_win_rate', 'last10_win_rate', 'last5_won', 'last5_finish_rate',
    'trend_score', 'opp_quality', 'layoff_days',
]
# qa_SLpM/qa_SApM excluded (8SI v2 Stage 0.2): dropped from the trainer's
# feature list (qa_SLpM == qa_win_rate, qa_SApM == 1 - qa_win_rate exactly —
# pure duplicates, see docs/V2_LOG.md), so build_dataset() no longer merges
# R/B_qa_SLpM/qa_SApM columns onto df at all — nothing to compare against.
# get_fighter_state_asof() still computes and returns them (general lookup,
# untouched), just not exercised by this parity test anymore.
QA_COLS = ['qa_win_rate', 'qa_finish_rate']
ELO_COLS = ['elo', 'elo_trend']


def _close(a, b, tol=TOL):
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, abs_tol=tol)


@pytest.mark.parametrize(
    'sample', samples,
    ids=lambda s: f"{s['R_fighter']}_v_{s['B_fighter']}@{pd.Timestamp(s['date']).date()}",
)
def test_train_serve_parity(sample):
    date = pd.Timestamp(sample['date'])
    key = f"{sample['R_fighter']}_v_{sample['B_fighter']}@{date.date()}"
    r_state, b_state = _states_by_sample[key]

    for corner, fighter_col, state in (('R', 'R_fighter', r_state), ('B', 'B_fighter', b_state)):
        fighter = sample[fighter_col]

        for col in CAREER_COLS + QA_COLS + ELO_COLS + STYLE_STATS:
            trainer_val = float(sample[f'{corner}_{col}'])
            serve_val   = float(state[col])
            assert _close(trainer_val, serve_val), (
                f'{corner}_{col} mismatch for {fighter} @ {date.date()}: '
                f'trainer={trainer_val} serve={serve_val}'
            )

        trainer_gf = float(sample[f'{corner}_got_finished_rate'])
        serve_gf   = float(state['got_finished_rate'])
        assert _close(trainer_gf, serve_gf), (
            f'{corner}_got_finished_rate mismatch for {fighter} @ {date.date()}: '
            f'trainer={trainer_gf} serve={serve_gf}'
        )
