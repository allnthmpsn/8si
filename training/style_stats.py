"""
As-of (leak-free) career style stats — SLpM, SApM, Str_Acc, Str_Def, TD_Avg,
TD_Acc, TD_Def, Sub_Avg — computed per fighter, per fight, using ONLY that
fighter's fights strictly before the fight's date.

Source: data/ufc_gold_dataset_final.csv (per-fight totals for both corners).
The plan originally pointed at data/ufc_stats_fights.csv, but that file only
has fighter/opponent/date/result/method/round/time — no strike or takedown
counts at all. ufc_gold_dataset_final.csv carries both fighters' landed/
attempted counts per fight, which is also what makes Str_Def/TD_Def
computable (they need the OPPONENT's attempts against this fighter).

Replaces the old ufc_fighters_final_updated.csv merge in train_model1.py,
which was a plain name-keyed join onto a fighter's CURRENT career snapshot —
every historical fight saw that fighter's future averages.
"""
import numpy as np
import pandas as pd

EPS = 1e-9

STYLE_STATS = ['SLpM', 'SApM', 'Str_Acc', 'Str_Def', 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']


def _to_long(fights_df):
    """One row per fight per corner: fighter's own stats + what the opponent
    landed/attempted against them (needed for the defensive rates)."""
    f1_cols = {
        'Fighter_1': 'fighter', 'Fighter_2': 'opponent',
        'F1_Sig_Landed': 'sig_landed', 'F1_Sig_Att': 'sig_att',
        'F2_Sig_Landed': 'opp_sig_landed', 'F2_Sig_Att': 'opp_sig_att',
        'F1_TD_Landed': 'td_landed', 'F1_TD_Att': 'td_att',
        'F2_TD_Landed': 'opp_td_landed', 'F2_TD_Att': 'opp_td_att',
        'F1_Sub_Att': 'sub_att',
    }
    f2_cols = {
        'Fighter_2': 'fighter', 'Fighter_1': 'opponent',
        'F2_Sig_Landed': 'sig_landed', 'F2_Sig_Att': 'sig_att',
        'F1_Sig_Landed': 'opp_sig_landed', 'F1_Sig_Att': 'opp_sig_att',
        'F2_TD_Landed': 'td_landed', 'F2_TD_Att': 'td_att',
        'F1_TD_Landed': 'opp_td_landed', 'F1_TD_Att': 'opp_td_att',
        'F2_Sub_Att': 'sub_att',
    }
    keep_extra = ['Event_Date', 'Total_Fight_Time_Sec']
    f1 = fights_df.rename(columns=f1_cols)[list(f1_cols.values()) + keep_extra]
    f2 = fights_df.rename(columns=f2_cols)[list(f2_cols.values()) + keep_extra]

    long_df = pd.concat([f1, f2], ignore_index=True)
    long_df = long_df.rename(columns={'Event_Date': 'date'})
    long_df['date'] = pd.to_datetime(long_df['date'])
    return long_df


def compute_style_stats_asof(fights_df):
    """
    fights_df: raw ufc_gold_dataset_final.csv-shaped DataFrame (one row per
    fight, F1_*/F2_* columns for both corners).

    Returns DataFrame[fighter, date, SLpM, SApM, Str_Acc, Str_Def, TD_Avg,
    TD_Acc, TD_Def, Sub_Avg] — one row per (fighter, fight), values computed
    from ONLY that fighter's fights strictly before `date` (shift(1)
    discipline, matching compute_career_stats in train_model1.py). NaN where
    the fighter has no prior fight in this source (their UFC debut, or a
    fighter-name spelling not present in ufc_gold_dataset_final.csv) —
    callers must impute this themselves rather than fillna(0), since 0 SLpM
    means "worst fighter alive," not "unknown."

    Observed-mask design: a raw column can be individually NaN for a given
    fight (this source has no producer script — see docs/DATA_SOURCES.md —
    so a future re-scrape could easily introduce gaps even though today's
    copy happens to have none). The naive `cumsum() - own_value` trick used
    elsewhere in this codebase breaks under that: pandas' cumsum reports
    NaN at the position of a NaN input even with skipna=True, so a single
    missing raw value would (a) wrongly NaN out THAT fight's own "prior"
    aggregate even when real prior data exists, and (b) silently and
    permanently undercount every LATER fight for the same fighter — treating
    the gap as a silent zero-contribution with no record it happened, not
    caught by any test since a leak test recomputes with the same function.

    Fixed per stat (not per raw column): each of the 8 stats is a
    numerator/denominator pair (e.g. SLpM = sig_landed / fight_min). A
    fight is only counted toward EITHER side of a given pair's running sum
    if BOTH columns were observed for that fight — a fight missing just the
    numerator doesn't get to silently count its minutes toward the
    denominator (which would bias the rate down), and vice versa. "Has
    data" is a per-pair cumulative COUNT of jointly-observed prior fights
    (never itself NaN, so immune to the poisoning above), tracked
    separately from the value sums.
    """
    df = _to_long(fights_df)
    df = df.sort_values(['fighter', 'date']).reset_index(drop=True)
    df['fight_min'] = df['Total_Fight_Time_Sec'] / 60.0

    # (stat, numerator col, denominator col, denominator is per-15-min, ratio is a complement (1 - x))
    PAIRS = [
        ('SLpM',    'sig_landed',     'fight_min',   False, False),
        ('SApM',    'opp_sig_landed', 'fight_min',   False, False),
        ('TD_Avg',  'td_landed',      'fight_min',   True,  False),
        ('Sub_Avg', 'sub_att',        'fight_min',   True,  False),
        ('Str_Acc', 'sig_landed',     'sig_att',     False, False),
        ('Str_Def', 'opp_sig_landed', 'opp_sig_att', False, True),
        ('TD_Acc',  'td_landed',      'td_att',      False, False),
        ('TD_Def',  'opp_td_landed',  'opp_td_att',  False, True),
    ]

    for stat, ncol, dcol, _, _ in PAIRS:
        joint = df[ncol].notna() & df[dcol].notna()
        df[f'_jn_{stat}'] = df[ncol].where(joint, 0.0)
        df[f'_jd_{stat}'] = df[dcol].where(joint, 0.0)
        df[f'_jo_{stat}'] = joint.astype(float)

    g = df.groupby('fighter', sort=False)
    for stat, *_ in PAIRS:
        df[f'_csn_{stat}'] = (g[f'_jn_{stat}'].cumsum() - df[f'_jn_{stat}']).astype(float)
        df[f'_csd_{stat}'] = (g[f'_jd_{stat}'].cumsum() - df[f'_jd_{stat}']).astype(float)
        df[f'_cso_{stat}'] = (g[f'_jo_{stat}'].cumsum() - df[f'_jo_{stat}']).astype(float)

    nan = np.nan
    for stat, ncol, dcol, per15, is_complement in PAIRS:
        # Two separate conditions, both required: _cso>0 ("we've actually
        # observed this pair before" — fixes the NaN-poisoning bug) AND
        # _csd>0 ("those observed fights collectively attempted at least
        # once" — a fighter whose priors are all genuinely zero attempts
        # has an undefined 0/0 rate, same as the pre-fix behavior; without
        # this second check a genuinely-zero denominator would silently
        # report 0% instead of NaN, which is wrong for e.g. a fighter who
        # has simply never attempted a takedown yet).
        has = (df[f'_cso_{stat}'] > 0).to_numpy() & (df[f'_csd_{stat}'] > 0).to_numpy()
        denom = df[f'_csd_{stat}'] / 15.0 if per15 else df[f'_csd_{stat}']
        ratio = df[f'_csn_{stat}'].to_numpy() / denom.clip(lower=EPS).to_numpy()
        if is_complement:
            ratio = 1.0 - ratio
        df[stat] = np.where(has, ratio, nan)

    return df[['fighter', 'date'] + STYLE_STATS]
