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
    """
    df = _to_long(fights_df)
    df = df.sort_values(['fighter', 'date']).reset_index(drop=True)
    df['fight_min'] = df['Total_Fight_Time_Sec'] / 60.0

    raw_cols = ['sig_landed', 'sig_att', 'opp_sig_landed', 'opp_sig_att',
                'td_landed', 'td_att', 'opp_td_landed', 'opp_td_att',
                'sub_att', 'fight_min']
    g = df.groupby('fighter', sort=False)
    for col in raw_cols:
        df[f'_cs_{col}'] = (g[col].cumsum() - df[col]).astype(float)

    has_min         = (df['_cs_fight_min'] > 0).to_numpy()
    has_sig_att     = (df['_cs_sig_att'] > 0).to_numpy()
    has_opp_sig_att = (df['_cs_opp_sig_att'] > 0).to_numpy()
    has_td_att      = (df['_cs_td_att'] > 0).to_numpy()
    has_opp_td_att  = (df['_cs_opp_td_att'] > 0).to_numpy()

    min_  = df['_cs_fight_min'].clip(lower=EPS).to_numpy()
    per15 = (df['_cs_fight_min'] / 15.0).clip(lower=EPS).to_numpy()

    nan = np.nan
    df['SLpM']    = np.where(has_min, df['_cs_sig_landed'] / min_, nan)
    df['SApM']    = np.where(has_min, df['_cs_opp_sig_landed'] / min_, nan)
    df['Str_Acc'] = np.where(has_sig_att, df['_cs_sig_landed'] / df['_cs_sig_att'].clip(lower=EPS), nan)
    df['Str_Def'] = np.where(has_opp_sig_att, 1.0 - df['_cs_opp_sig_landed'] / df['_cs_opp_sig_att'].clip(lower=EPS), nan)
    df['TD_Avg']  = np.where(has_min, df['_cs_td_landed'] / per15, nan)
    df['Sub_Avg'] = np.where(has_min, df['_cs_sub_att'] / per15, nan)
    df['TD_Acc']  = np.where(has_td_att, df['_cs_td_landed'] / df['_cs_td_att'].clip(lower=EPS), nan)
    df['TD_Def']  = np.where(has_opp_td_att, 1.0 - df['_cs_opp_td_landed'] / df['_cs_opp_td_att'].clip(lower=EPS), nan)

    return df[['fighter', 'date'] + STYLE_STATS]
