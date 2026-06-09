from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import os
import re
import requests
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from difflib import SequenceMatcher

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Load models (LR 70% + XGB 30% blend, 72.81% temporal accuracy, 129 features)
# V2: men's fights only, recency HL=730d, expanded 2015+ window, QA stats
# ─────────────────────────────────────────────────────────────────────────────
model_lr  = joblib.load('../model/ufc_model_best.pkl')
model_xgb = joblib.load('../model/ufc_model_xgb.pkl')
feature_columns = joblib.load('../model/feature_columns_best.pkl')

LR_WEIGHT  = 0.70
XGB_WEIGHT = 0.30

with open('../model/model_metadata.json') as _f:
    metadata = json.load(_f)

# ── WC stat averages for zero-stat fallback ──
try:
    with open('../model/wc_stat_averages.json') as _wf:
        _wc_stat_avgs = json.load(_wf)
except Exception:
    _wc_stat_avgs = {}

# Map weight class name → lbs for CSV lookup
_WC_NAME_TO_LBS = {
    "Women's Strawweight": 115, "Women's Flyweight": 125,
    "Women's Bantamweight": 135, "Women's Featherweight": 145,
    "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
    "Lightweight": 155, "Welterweight": 170, "Middleweight": 185,
    "Light Heavyweight": 205, "Heavyweight": 265,
}


# ─────────────────────────────────────────────────────────────────────────────
# Load Model 2 (50% LR + 50% XGB blend, 42 features) + ROI optimizer config
# ─────────────────────────────────────────────────────────────────────────────
M2_LR_WEIGHT  = 0.50
M2_XGB_WEIGHT = 0.50

try:
    model2          = joblib.load('../model/ufc_model2a_best.pkl')
    model2_xgb      = joblib.load('../model/ufc_model2a_xgb.pkl')
    model2_features = joblib.load('../model/ufc_model2a_features.pkl')
    with open('../model/model2a_tier_stats.json') as _f:
        _m2_tier_stats = json.load(_f)
    with open('../model/roi_optimizer.json') as _f:
        roi_config = json.load(_f)
    _M2_READY = True
    print(f"Model 2A: loaded ({len(model2_features)} features, 50/50 LR+XGB blend)")
except Exception as _e:
    model2 = model2_xgb = model2_features = roi_config = _m2_tier_stats = None
    _M2_READY = False
    print(f"Model 2A: not available ({_e})")

try:
    _model2b_rf       = joblib.load('../model/ufc_model2b.pkl')
    _model2b_features = joblib.load('../model/ufc_model2b_features.pkl')
    with open('../model/ufc_model2b_config.json') as _f:
        _model2b_config = json.load(_f)
    _M2B_READY = True
    print(f"Model 2B: loaded (RF, {len(_model2b_features)} features, SPLIT floor {_model2b_config['split_floor']})")
except Exception as _e:
    _model2b_rf = _model2b_features = _model2b_config = None
    _M2B_READY = False
    print(f"Model 2B: not available ({_e})")

try:
    model3a_lr       = joblib.load('../model/ufc_model3a_lr.pkl')
    model3a_xgb      = joblib.load('../model/ufc_model3a_xgb.pkl')
    model3a_features = joblib.load('../model/ufc_model3a_features.pkl')
    model3b_rf       = joblib.load('../model/ufc_model3b_rf.pkl')
    model3b_xgb      = joblib.load('../model/ufc_model3b_xgb.pkl')
    model3b_features = joblib.load('../model/ufc_model3b_features.pkl')
    MODEL3_AVAILABLE = True
    print(f"Model 3A: loaded ({len(model3a_features)} features, 30/70 LR+XGB, 64.94% acc)")
    print(f"Model 3B: loaded ({len(model3b_features)} features, 40% RF+60% XGB, 46.56% six-class, 70.67% dir)")
except Exception as _e:
    model3a_lr = model3a_xgb = model3a_features = None
    model3b_rf = model3b_xgb = model3b_features = None
    MODEL3_AVAILABLE = False
    print(f"Model 3A/3B: not available ({_e})")

GAP_THRESHOLD  = 0.10   # 10% — statistically robust (168 bets, +34.4% ROI)
KELLY_FRACTION = 1 / 3
MAX_BET        = 100
BANKROLL       = 1000

# ─────────────────────────────────────────────────────────────────────────────
# Load women's models (separate from men's — will not overwrite men's model vars)
# ─────────────────────────────────────────────────────────────────────────────
try:
    womens_model_lr  = joblib.load('../model/ufc_model_womens_lr.pkl')
    womens_model_xgb = joblib.load('../model/ufc_model_womens_xgb.pkl')
    womens_features  = joblib.load('../model/ufc_model_womens_features.pkl')
    with open('../model/ufc_model_womens_metadata.json') as _f:
        _womens_metadata = json.load(_f)
    WOMENS_MODEL_AVAILABLE = True
    _womens_acc = _womens_metadata['temporal_accuracy']
    print(f"Women's M1: loaded ({len(womens_features)} features, 70/30 LR+XGB, {_womens_acc*100:.2f}%)")
except Exception as _e:
    womens_model_lr = womens_model_xgb = womens_features = _womens_metadata = None
    WOMENS_MODEL_AVAILABLE = False
    _womens_acc = 0.0
    print(f"Women's M1: not available ({_e})")

try:
    womens_m2a_lr       = joblib.load('../model/ufc_model2a_womens_lr.pkl')
    womens_m2a_xgb      = joblib.load('../model/ufc_model2a_womens_xgb.pkl')
    womens_m2a_features = joblib.load('../model/ufc_model2a_womens_features.pkl')
    with open('../model/ufc_model2a_womens_metadata.json') as _f:
        _womens_m2a_metadata = json.load(_f)
    WOMENS_M2A_AVAILABLE = True
    _womens_m2a_acc = _womens_m2a_metadata.get('temporal_accuracy', 0.0)
    print(f"Women's M2A: loaded ({len(womens_m2a_features)} features, 50/50 LR+XGB, {_womens_m2a_acc*100:.2f}%)")
except Exception as _e:
    womens_m2a_lr = womens_m2a_xgb = womens_m2a_features = _womens_m2a_metadata = None
    WOMENS_M2A_AVAILABLE = False
    _womens_m2a_acc = 0.0
    print(f"Women's M2A: not available ({_e})")

# ─────────────────────────────────────────────────────────────────────────────
# Load datasets
# ─────────────────────────────────────────────────────────────────────────────
df_fighters = pd.read_csv('../data/ufc-master.csv', low_memory=False)
df_fighters['date'] = pd.to_datetime(df_fighters['date'])

with open('../data/sherdog_records.pkl', 'rb') as f:
    sherdog_records = pickle.load(f)

career_df = pd.read_csv('../data/career_fights_updated.csv')
career_df['date'] = pd.to_datetime(career_df['date'])
career_df = career_df.sort_values(['fighter', 'date']).reset_index(drop=True)

fighters_stats_df = pd.read_csv('../data/ufc_fighters_final_updated.csv')
for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
    fighters_stats_df[col] = pd.to_numeric(
        fighters_stats_df[col].astype(str).str.replace('%', '', regex=False),
        errors='coerce'
    ).fillna(0) / 100.0

# ─────────────────────────────────────────────────────────────────────────────
# Build lookup tables at startup
# ─────────────────────────────────────────────────────────────────────────────
def _parse_height_cm(val):
    """Accept cm float (180.34) or feet/inches string (5'10\") → float cm."""
    if not val or str(val).strip() in ('', 'nan', 'NaN', '0', '0.0'):
        return 175.0
    try:
        v = float(val)
        return v if v > 50 else 175.0   # sanity: must be > 50 cm
    except (ValueError, TypeError):
        pass
    m = re.match(r"(\d+)'[\"']?\s*(\d+)", str(val))
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 2)
    return 175.0


def _safe_float(val, default=0.0):
    try:
        s = str(val).strip().replace('--', '').replace('"', '')
        return float(s) if s else default
    except (ValueError, TypeError):
        return default


fighter_stats_lookup = {}
for _, row in fighters_stats_df.iterrows():
    name = str(row['Fighter_Name'])
    fighter_stats_lookup[name] = {
        'SLpM':       _safe_float(row.get('SLpM',    0)),
        'SApM':       _safe_float(row.get('SApM',    0)),
        'Str_Acc':    float(row.get('Str_Acc', 0) or 0),
        'Str_Def':    float(row.get('Str_Def', 0) or 0),
        'TD_Avg':     _safe_float(row.get('TD_Avg',  0)),
        'TD_Acc':     float(row.get('TD_Acc', 0) or 0),
        'TD_Def':     float(row.get('TD_Def', 0) or 0),
        'Sub_Avg':    _safe_float(row.get('Sub_Avg', 0)),
        'Reach':      _safe_float(row.get('Reach',   '')),
        'Height_cms': _parse_height_cm(row.get('Height', '')),
        'Stance':     str(row.get('Stance', '') or ''),
        'DOB':        str(row.get('DOB', '') or ''),
        'wins':       int(_safe_float(row.get('Wins',   0))),
        'losses':     int(_safe_float(row.get('Losses', 0))),
    }

# Pre-compute career win rate for every fighter (used for opp_quality lookup)
_fighter_wr: dict[str, float] = {}
for _name, _grp in career_df.groupby('fighter'):
    _n = len(_grp)
    _w = int((_grp['won'] == 1).sum())
    _fighter_wr[_name] = _w / _n if _n > 0 else 0.5

# ── Elo ratings ──────────────────────────────────────────────────────────────
_elo_curr_df = pd.read_csv('../data/elo_current.csv')
_elo_hist_df = pd.read_csv('../data/elo_ratings_history.csv')
_elo_hist_df['date'] = pd.to_datetime(_elo_hist_df['date'])
_elo_hist_df = _elo_hist_df.sort_values(['fighter', 'date']).reset_index(drop=True)

_elo_curr_map = dict(zip(_elo_curr_df['fighter'], _elo_curr_df['current_elo']))
_elo_lookup: dict[str, dict] = {}
for _fighter, _grp in _elo_hist_df.groupby('fighter'):
    _grp_s = _grp.sort_values('date')
    _cur   = float(_elo_curr_map.get(_fighter, 1500.0))
    # elo_trend for next fight = current_elo - elo_before 3 fights ago
    _trend = float(_cur - _grp_s['elo_before'].iloc[-3]) if len(_grp_s) >= 3 else 0.0
    _elo_lookup[_fighter] = {'elo': _cur, 'elo_trend': _trend}

# ── Women's Elo (computed from women's fights only — separate from men's pool) ──
def _compute_womens_elo(df_master, K=48, base=1500.0):
    _WOMENS_CLS = {
        "Women's Strawweight", "Women's Flyweight",
        "Women's Bantamweight", "Women's Featherweight",
    }
    rows = df_master[
        df_master['weight_class'].isin(_WOMENS_CLS) &
        df_master['Winner'].isin(['Red', 'Blue'])
    ].sort_values('date').reset_index(drop=True)
    elo, history = {}, []
    for _, row in rows.iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        r_b = elo.get(r, base)
        b_b = elo.get(b, base)
        r_exp = 1.0 / (1.0 + 10.0 ** ((b_b - r_b) / 400.0))
        r_act, b_act = (1.0, 0.0) if row['Winner'] == 'Red' else (0.0, 1.0)
        r_a = r_b + K * (r_act - r_exp)
        b_a = b_b + K * (b_act - (1 - r_exp))
        history.append({'fighter': r, 'date': row['date'], 'elo_before': r_b})
        history.append({'fighter': b, 'date': row['date'], 'elo_before': b_b})
        elo[r] = r_a
        elo[b] = b_a
    hist = pd.DataFrame(history).sort_values(['fighter', 'date']).reset_index(drop=True)
    lookup = {}
    for ftr, grp in hist.groupby('fighter'):
        cur = float(elo.get(ftr, base))
        trend = float(cur - grp['elo_before'].iloc[-3]) if len(grp) >= 3 else 0.0
        lookup[ftr] = {'elo': cur, 'elo_trend': trend}
    return lookup

_womens_elo_lookup = _compute_womens_elo(df_fighters)
print(f"Women's Elo: computed for {len(_womens_elo_lookup)} fighters")

print(f"LR model: {type(model_lr).__name__}, XGB: {type(model_xgb).__name__}, features={len(feature_columns)}, blend={LR_WEIGHT}/{XGB_WEIGHT} (72.81% temporal, men's only)")
print(f"Fighter stats: {len(fighter_stats_lookup)}, Career WR cache: {len(_fighter_wr)}")

# ─────────────────────────────────────────────────────────────────────────────
# Weight class ordinal  (matches training script exactly)
# ─────────────────────────────────────────────────────────────────────────────
WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11, "Catch Weight": 6,
}

WOMENS_CLASSES = {
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
}

WC_ORDER_WOMENS = {
    "Women's Strawweight":   0,
    "Women's Flyweight":     1,
    "Women's Bantamweight":  2,
    "Women's Featherweight": 3,
}


def is_womens_fight(weight_class: str) -> bool:
    return weight_class in WOMENS_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def clean_val(val, default=0):
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    if hasattr(val, 'item'):
        return val.item()
    return val


def _layoff_buckets(days: float) -> dict:
    return {
        'lt90':    1 if days < 90 else 0,
        '90_180':  1 if 90 <= days < 180 else 0,
        '180_365': 1 if 180 <= days < 365 else 0,
        'gt365':   1 if days >= 365 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_latest_ufc_stats(name: str) -> dict | None:
    as_red  = df_fighters[df_fighters['R_fighter'] == name].sort_values('date', ascending=False)
    as_blue = df_fighters[df_fighters['B_fighter'] == name].sort_values('date', ascending=False)

    red_date  = as_red.iloc[0]['date']  if len(as_red)  > 0 else pd.Timestamp.min
    blue_date = as_blue.iloc[0]['date'] if len(as_blue) > 0 else pd.Timestamp.min

    if red_date >= blue_date and len(as_red) > 0:
        row, p = as_red.iloc[0], 'R'
    elif len(as_blue) > 0:
        row, p = as_blue.iloc[0], 'B'
    else:
        return None

    return {
        'wins':                      clean_val(row.get(f'{p}_wins', 0)),
        'losses':                    clean_val(row.get(f'{p}_losses', 0)),
        'avg_SIG_STR_landed':        clean_val(row.get(f'{p}_avg_SIG_STR_landed', 0)),
        'avg_SIG_STR_pct':           clean_val(row.get(f'{p}_avg_SIG_STR_pct', 0)),
        'avg_TD_landed':             clean_val(row.get(f'{p}_avg_TD_landed', 0)),
        'avg_TD_pct':                clean_val(row.get(f'{p}_avg_TD_pct', 0)),
        'avg_SUB_ATT':               clean_val(row.get(f'{p}_avg_SUB_ATT', 0)),
        'win_by_KO_TKO':             clean_val(row.get(f'{p}_win_by_KO/TKO', 0)),
        'win_by_Submission':         clean_val(row.get(f'{p}_win_by_Submission', 0)),
        'win_by_Decision_Unanimous': clean_val(row.get(f'{p}_win_by_Decision_Unanimous', 0)),
        'win_by_Decision_Split':     clean_val(row.get(f'{p}_win_by_Decision_Split', 0)),
        'win_by_Decision_Majority':  clean_val(row.get(f'{p}_win_by_Decision_Majority', 0)),
        'Height_cms':                clean_val(row.get(f'{p}_Height_cms', 175)),
        'Reach_cms':                 clean_val(row.get(f'{p}_Reach_cms', 175)),
        'age':                       clean_val(row.get(f'{p}_age', 28)),
        'Stance':                    str(row.get(f'{p}_Stance', 'Orthodox') or 'Orthodox'),
        'current_win_streak':        clean_val(row.get(f'{p}_current_win_streak', 0)),
        'current_lose_streak':       clean_val(row.get(f'{p}_current_lose_streak', 0)),
        'longest_win_streak':        clean_val(row.get(f'{p}_longest_win_streak', 0)),
        'total_title_bouts':         clean_val(row.get(f'{p}_total_title_bouts', 0)),
    }


def get_sherdog_total_record(name: str) -> tuple[int | None, int | None]:
    if name not in sherdog_records:
        return None, None
    fights = sherdog_records[name]['fights']
    wins   = sum(1 for f in fights if f['result'] == 'win'  and f['date'] is not None)
    losses = sum(1 for f in fights if f['result'] == 'loss' and f['date'] is not None)
    return wins, losses


def get_career_stats(name: str) -> dict:
    """Return all career rolling stats for a fighter as of today (all fights to date)."""
    ff = career_df[career_df['fighter'] == name].sort_values('date')

    defaults = {
        'cum_fights': 0, 'career_win_rate': 0.5,
        'ko_finish_rate': 0.0, 'sub_finish_rate': 0.0,
        'last3_win_rate': 0.5, 'last5_won': 0.5,
        'last10_win_rate': 0.5, 'last5_finish_rate': 0.3,
        'trend_score': 0.0, 'layoff_days': 180.0, 'opp_quality': 0.5,
    }

    if len(ff) == 0:
        return defaults

    total = len(ff)
    wins  = int((ff['won'] == 1).sum())

    did_ko  = ((ff['won'] == 1) & ff['method'].str.contains('KO|TKO', case=False, na=False)).sum()
    did_sub = ((ff['won'] == 1) & ff['method'].str.contains('Sub|Submission', case=False, na=False)).sum()

    last3  = ff.tail(3)
    last5  = ff.tail(5)
    last10 = ff.tail(10)

    last3_win_rate  = float((last3['won']  == 1).mean()) if len(last3)  > 0 else 0.5
    last5_won       = float((last5['won']  == 1).mean()) if len(last5)  > 0 else 0.5
    last10_win_rate = float((last10['won'] == 1).mean()) if len(last10) > 0 else 0.5

    last5_finish = float(
        ((last5['won'] == 1) &
         last5['method'].str.contains('KO|TKO|Sub|Submission', case=False, na=False)).mean()
    ) if len(last5) > 0 else 0.0

    # Opponent quality: avg career win rate of last-5 opponents
    last5_opps    = ff.tail(5)['opponent'].tolist()
    opp_win_rates = [_fighter_wr[opp] for opp in last5_opps if opp in _fighter_wr]
    opp_quality   = float(np.mean(opp_win_rates)) if opp_win_rates else 0.5

    last_date  = ff['date'].max()
    layoff_days = float((pd.Timestamp.now() - last_date).days)

    return {
        'cum_fights':       total,
        'career_win_rate':  float(wins / total) if total > 0 else 0.5,
        'ko_finish_rate':   float(did_ko  / total) if total > 0 else 0.0,
        'sub_finish_rate':  float(did_sub / total) if total > 0 else 0.0,
        'last3_win_rate':   last3_win_rate,
        'last5_won':        last5_won,
        'last10_win_rate':  last10_win_rate,
        'last5_finish_rate': last5_finish,
        'trend_score':      last3_win_rate - last10_win_rate,
        'layoff_days':      layoff_days,
        'opp_quality':      opp_quality,
    }


def get_fighter_extra_stats(name: str, weight_class: str = '') -> dict:
    stats = fighter_stats_lookup.get(name, {})

    dob_str = stats.get('DOB', '')
    age = None
    if dob_str and dob_str.lower() not in ('nan', ''):
        try:
            age = (datetime.now() - pd.to_datetime(dob_str)).days // 365
        except Exception:
            pass

    # Weight-class average fallback for experienced fighters with zero striking stats.
    # True debutants (no UFC history) keep zeros — the frontend shows N/A for them.
    ufc_row_r = df_fighters[df_fighters['R_fighter'] == name]
    ufc_row_b = df_fighters[df_fighters['B_fighter'] == name]
    # is_debut: ONLY true when fighter has zero UFC fight history — not just 0 wins
    is_debut = (len(ufc_row_r) == 0) and (len(ufc_row_b) == 0)

    ufc_wins = 0
    ufc_row = ufc_row_r if len(ufc_row_r) > 0 else ufc_row_b
    if len(ufc_row) > 0:
        latest = ufc_row.sort_values('date', ascending=False).iloc[0]
        p = 'R' if latest.get('R_fighter') == name else 'B'
        ufc_wins = int(latest.get(f'{p}_wins', 0) or 0)
    wc_avgs = _wc_stat_avgs.get(weight_class, {})

    def _stat(key, raw_val):
        v = float(raw_val or 0)
        if v == 0 and ufc_wins > 0 and key in wc_avgs:
            return float(wc_avgs[key])
        return v

    return {
        'SLpM':        _stat('SLpM',    stats.get('SLpM',    0)),
        'SApM':        _stat('SApM',    stats.get('SApM',    0)),
        'Str_Acc':     _stat('Str_Acc', stats.get('Str_Acc', 0)),
        'Str_Def':     _stat('Str_Def', stats.get('Str_Def', 0)),
        'TD_Avg':      _stat('TD_Avg',  stats.get('TD_Avg',  0)),
        'TD_Acc':      _stat('TD_Acc',  stats.get('TD_Acc',  0)),
        'TD_Def':      _stat('TD_Def',  stats.get('TD_Def',  0)),
        'Sub_Avg':     _stat('Sub_Avg', stats.get('Sub_Avg', 0)),
        'Reach':       stats.get('Reach', 0),
        'Stance':      stats.get('Stance', ''),
        'is_southpaw': 1 if stats.get('Stance', '').lower() == 'southpaw' else 0,
        'age':         age,
        'is_debut':    is_debut,
    }


def get_elo_stats(name: str) -> dict:
    stats = _elo_lookup.get(name, {})
    return {
        'elo':       float(stats.get('elo', 1500.0)),
        'elo_trend': float(stats.get('elo_trend', 0.0)),
    }


def get_womens_elo_stats(name: str) -> dict:
    stats = _womens_elo_lookup.get(name, {})
    return {
        'elo':       float(stats.get('elo', 1500.0)),
        'elo_trend': float(stats.get('elo_trend', 0.0)),
    }


_M3_LOW_CONF_DIVISIONS = {"Women's Flyweight", "Light Heavyweight", "Bantamweight"}


def get_career_method_rates(name: str) -> dict:
    """Compute career method/finish rates for Model 3A/3B inference."""
    ff = career_df[career_df['fighter'] == name].sort_values('date')

    defaults = {
        'career_is_finish': 0.0, 'career_is_decision': 0.5,
        'career_is_ko': 0.0,     'career_is_sub': 0.0,
        'career_finish_delivered': 0.0, 'career_finish_received': 0.0,
        'career_dec_delivered': 0.0,    'career_dec_received': 0.0,
        'career_n_fights': 0,
    }

    if len(ff) == 0:
        return defaults

    n = len(ff)
    method = ff['method'].fillna('')
    won    = ff['won'].fillna(0).astype(int)

    is_finish  = (method.str.contains('KO|TKO|Sub|Submission', case=False)).astype(int)
    is_dec     = (method.str.contains('Decision', case=False)).astype(int)
    is_ko      = (method.str.contains('KO|TKO', case=False)).astype(int)
    is_sub     = (method.str.contains('Sub|Submission', case=False)).astype(int)

    fin_del = ((won == 1) & (is_finish == 1)).astype(int)
    fin_rec = ((won == 0) & (is_finish == 1)).astype(int)
    dec_del = ((won == 1) & (is_dec == 1)).astype(int)
    dec_rec = ((won == 0) & (is_dec == 1)).astype(int)

    return {
        'career_is_finish':        float(is_finish.mean()),
        'career_is_decision':      float(is_dec.mean()),
        'career_is_ko':            float(is_ko.mean()),
        'career_is_sub':           float(is_sub.mean()),
        'career_finish_delivered': float(fin_del.mean()),
        'career_finish_received':  float(fin_rec.mean()),
        'career_dec_delivered':    float(dec_del.mean()),
        'career_dec_received':     float(dec_rec.mean()),
        'career_n_fights':         n,
    }


def _ml_to_impl_prob(ml: float) -> float:
    """American moneyline → implied probability. Returns 0.5 for missing (0)."""
    if ml == 0:
        return 0.5
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "8 Sided Insights API running",
        "model1_mens": "LR 70% + XGB 30% blend (72.81% temporal accuracy, 129 features, men's only, recency HL=730d)",
        "model1_womens": (
            f"LR 70% + XGB 30% blend ({_womens_acc*100:.2f}% temporal accuracy, "
            f"{len(womens_features)} features, women's only, baseline v1)"
        ) if WOMENS_MODEL_AVAILABLE else "not loaded",
        "model2a_womens": (
            f"LR 50% + XGB 50% + odds blend ({_womens_m2a_acc*100:.2f}% accuracy, "
            f"{len(womens_m2a_features)} features)"
        ) if WOMENS_M2A_AVAILABLE else "not loaded",
        "model2_mens": "50% LR + 50% XGB blend (73.20% temporal accuracy, 42 features, key: tier_hist_win_rate)" if _M2_READY else "not loaded",
        "model3a": "30% LR + 70% XGB (64.94% accuracy, 63 features, Goes the Distance)" if MODEL3_AVAILABLE else "not loaded",
        "model3b": "40% RF + 60% XGB (46.56% six-class / 70.67% direction, 102 features, Winner+Method)" if MODEL3_AVAILABLE else "not loaded",
        "routing": "weight_class → womens model for women's divisions, mens model otherwise",
        "roi_settings": {
            "gap_threshold": f"{int(GAP_THRESHOLD * 100)}%",
            "kelly_fraction": f"1/{int(1 / KELLY_FRACTION)}",
            "max_bet":  f"${MAX_BET}",
            "bankroll": f"${BANKROLL}",
        } if _M2_READY else None,
    }


@app.get("/fighters")
def get_fighters():
    red  = df_fighters['R_fighter'].dropna().unique().tolist()
    blue = df_fighters['B_fighter'].dropna().unique().tolist()
    return {"fighters": sorted(set(red) | set(blue))}


@app.get("/fighter/{name}")
def get_fighter(name: str):
    ufc_stats = get_latest_ufc_stats(name)

    # Debut fighters have no UFC fights in ufc-master.csv yet — build from career data
    if ufc_stats is None:
        _extra  = get_fighter_extra_stats(name)
        _career = get_career_stats(name)
        _tw, _tl = get_sherdog_total_record(name)
        _row = fighter_stats_lookup.get(name, {})
        if _tw is None and _career['cum_fights'] == 0 and not _row:
            return {"error": "Fighter not found"}
        _h   = _row.get('Height_cms', 175.0)
        _r   = _row.get('Reach', 0) or _h   # fall back to height if reach unknown
        ufc_stats = {
            'wins': 0, 'losses': 0,
            'avg_SIG_STR_landed': 0, 'avg_SIG_STR_pct': 0,
            'avg_TD_landed': 0, 'avg_TD_pct': 0, 'avg_SUB_ATT': 0,
            'win_by_KO_TKO': 0, 'win_by_Submission': 0,
            'win_by_Decision_Unanimous': 0, 'win_by_Decision_Split': 0,
            'win_by_Decision_Majority': 0,
            'Height_cms': _h, 'Reach_cms': _r,
            'age': _extra.get('age') or 28,
            'Stance': _extra.get('Stance') or 'Orthodox',
            'current_win_streak': 0, 'current_lose_streak': 0,
            'longest_win_streak': 0, 'total_title_bouts': 0,
        }

    career  = get_career_stats(name)
    extra   = get_fighter_extra_stats(name)

    # Total MMA record from ufc_fighters_final_updated.csv (overrides UFC-only counts)
    _csv_row = fighter_stats_lookup.get(name, {})
    _csv_wins   = _csv_row.get('wins',   None)
    _csv_losses = _csv_row.get('losses', None)
    ufc_stats['total_wins']   = _csv_wins   if _csv_wins   is not None else ufc_stats['wins']
    ufc_stats['total_losses'] = _csv_losses if _csv_losses is not None else ufc_stats['losses']

    # Override stale CSV age with DOB-calculated age
    if extra.get('age') is not None:
        ufc_stats['age'] = extra['age']

    # Career stats
    ufc_stats['cum_fights']        = career['cum_fights']
    ufc_stats['career_win_rate']   = round(career['career_win_rate'],   3)
    ufc_stats['ko_finish_rate']    = round(career['ko_finish_rate'],    3)
    ufc_stats['sub_finish_rate']   = round(career['sub_finish_rate'],   3)
    ufc_stats['last3_win_rate']    = round(career['last3_win_rate'],    3)
    ufc_stats['last5_won']         = round(career['last5_won'],         3)
    ufc_stats['last10_win_rate']   = round(career['last10_win_rate'],   3)
    ufc_stats['last5_finish_rate'] = round(career['last5_finish_rate'], 3)
    ufc_stats['trend_score']       = round(career['trend_score'],       3)
    ufc_stats['layoff_days']       = career['layoff_days']
    ufc_stats['opp_quality']       = round(career['opp_quality'],       3)
    # kept for backwards compat
    ufc_stats['days_since_last']   = career['layoff_days']
    ufc_stats['fight_frequency']   = None  # not used in new model

    # Style stats
    ufc_stats['SLpM']        = extra['SLpM']
    ufc_stats['SApM']        = extra['SApM']
    ufc_stats['Str_Acc']     = extra['Str_Acc']
    ufc_stats['Str_Def']     = extra['Str_Def']
    ufc_stats['TD_Avg']      = extra['TD_Avg']
    ufc_stats['TD_Acc']      = extra['TD_Acc']
    ufc_stats['TD_Def']      = extra['TD_Def']
    ufc_stats['Sub_Avg']     = extra['Sub_Avg']
    ufc_stats['is_southpaw'] = extra['is_southpaw']
    ufc_stats['is_debut']    = extra.get('is_debut', False)

    # Elo ratings
    elo = get_elo_stats(name)
    ufc_stats['elo']       = round(elo['elo'], 1)
    ufc_stats['elo_trend'] = round(elo['elo_trend'], 1)

    return ufc_stats


# ─────────────────────────────────────────────────────────────────────────────
# FightInput — all 108-feature fields with sensible defaults
# ─────────────────────────────────────────────────────────────────────────────
class FightInput(BaseModel):
    # ── Fighter 1 (maps to Red corner) ──────────────────────────────────────
    F1_wins: float = 0
    F1_losses: float = 0
    F1_total_wins: float = 0
    F1_total_losses: float = 0
    F1_Height_cms: float = 175
    F1_Reach_cms: float = 175
    F1_age: float = 28
    F1_avg_SIG_STR_landed: float = 0
    F1_avg_SIG_STR_pct: float = 0
    F1_avg_TD_landed: float = 0
    F1_avg_TD_pct: float = 0
    F1_avg_SUB_ATT: float = 0
    F1_win_by_KO_TKO: float = 0
    F1_win_by_Submission: float = 0
    F1_win_by_Decision_Unanimous: float = 0
    F1_win_by_Decision_Split: float = 0
    F1_win_by_Decision_Majority: float = 0
    F1_current_win_streak: float = 0
    F1_current_lose_streak: float = 0
    F1_longest_win_streak: float = 0
    F1_total_title_bouts: float = 0
    F1_is_southpaw: float = 0
    # Career stats
    F1_cum_fights: float = 0
    F1_career_win_rate: float = 0.5
    F1_last5_won: float = 0.5
    F1_last5_finish_rate: float = 0.3
    F1_ko_finish_rate: float = 0.0
    F1_sub_finish_rate: float = 0.0
    F1_last3_win_rate: float = 0.5
    F1_last10_win_rate: float = 0.5
    F1_trend_score: float = 0.0
    F1_opp_quality: float = 0.5
    F1_layoff_days: float = 180.0
    # Style
    F1_SLpM: float = 0
    F1_SApM: float = 0
    F1_Str_Acc: float = 0
    F1_Str_Def: float = 0
    F1_TD_Avg: float = 0
    F1_TD_Acc: float = 0
    F1_TD_Def: float = 0
    F1_Sub_Avg: float = 0
    # ── Fighter 2 (maps to Blue corner) ─────────────────────────────────────
    F2_wins: float = 0
    F2_losses: float = 0
    F2_total_wins: float = 0
    F2_total_losses: float = 0
    F2_Height_cms: float = 175
    F2_Reach_cms: float = 175
    F2_age: float = 28
    F2_avg_SIG_STR_landed: float = 0
    F2_avg_SIG_STR_pct: float = 0
    F2_avg_TD_landed: float = 0
    F2_avg_TD_pct: float = 0
    F2_avg_SUB_ATT: float = 0
    F2_win_by_KO_TKO: float = 0
    F2_win_by_Submission: float = 0
    F2_win_by_Decision_Unanimous: float = 0
    F2_win_by_Decision_Split: float = 0
    F2_win_by_Decision_Majority: float = 0
    F2_current_win_streak: float = 0
    F2_current_lose_streak: float = 0
    F2_longest_win_streak: float = 0
    F2_total_title_bouts: float = 0
    F2_is_southpaw: float = 0
    # Career stats
    F2_cum_fights: float = 0
    F2_career_win_rate: float = 0.5
    F2_last5_won: float = 0.5
    F2_last5_finish_rate: float = 0.3
    F2_ko_finish_rate: float = 0.0
    F2_sub_finish_rate: float = 0.0
    F2_last3_win_rate: float = 0.5
    F2_last10_win_rate: float = 0.5
    F2_trend_score: float = 0.0
    F2_opp_quality: float = 0.5
    F2_layoff_days: float = 180.0
    # Style
    F2_SLpM: float = 0
    F2_SApM: float = 0
    F2_Str_Acc: float = 0
    F2_Str_Def: float = 0
    F2_TD_Avg: float = 0
    F2_TD_Acc: float = 0
    F2_TD_Def: float = 0
    F2_Sub_Avg: float = 0
    # ── Fight-level metadata ─────────────────────────────────────────────────
    weight_class: str = "Welterweight"
    title_bout: bool = False
    # ── Elo ratings ──────────────────────────────────────────────────────────
    F1_elo: float = 1500.0
    F2_elo: float = 1500.0
    F1_elo_trend: float = 0.0
    F2_elo_trend: float = 0.0
    # ── Optional odds for M2A prediction ─────────────────────────────────────
    f1_odds: float | None = None  # American ML (e.g. -150 or +130)
    f2_odds: float | None = None
    f1_dec_odds: float = 0
    f2_dec_odds: float = 0
    f1_ko_odds: float = 0
    f2_ko_odds: float = 0
    f1_sub_odds: float = 0
    f2_sub_odds: float = 0
    no_of_rounds: int = 3
    # ── Fighter names (optional — used by /method for outcome labels) ─────────
    f1_name: str | None = None
    f2_name: str | None = None
    # ── Legacy fields (kept for backwards compatibility) ─────────────────────
    F1_pre_ufc_wins: float = 0
    F1_pre_ufc_losses: float = 0
    F1_days_since_last: float = 180
    F1_fight_frequency: float = 2.0
    F2_pre_ufc_wins: float = 0
    F2_pre_ufc_losses: float = 0
    F2_days_since_last: float = 180
    F2_fight_frequency: float = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# /predict
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(fight: FightInput):
    d = fight.dict()

    # Stance
    R_southpaw  = int(d['F1_is_southpaw'])
    B_southpaw  = int(d['F2_is_southpaw'])
    orth_clash  = 1 if (R_southpaw == 0 and B_southpaw == 0) else 0
    south_clash = 1 if (R_southpaw == 1 and B_southpaw == 1) else 0

    # Age × experience interaction
    R_age_x_exp = d['F1_age'] * d['F1_cum_fights']
    B_age_x_exp = d['F2_age'] * d['F2_cum_fights']

    # Layoff buckets
    R_lb = _layoff_buckets(d['F1_layoff_days'])
    B_lb = _layoff_buckets(d['F2_layoff_days'])

    # Weight class ordinal and title bout binary
    wc_name = d['weight_class']
    _is_womens = is_womens_fight(wc_name)
    weight_class_ord = (
        WC_ORDER_WOMENS.get(wc_name, 1) if _is_womens else WC_ORDER.get(wc_name, 6)
    )
    title_bout_bin   = 1 if d['title_bout'] else 0

    data = {
        # ── Master base ───────────────────────────────────────────────────────
        'R_wins':                 d['F1_wins'],
        'R_losses':               d['F1_losses'],
        'R_Height_cms':           d['F1_Height_cms'],
        'R_age':                  d['F1_age'],
        'R_avg_SIG_STR_landed':   d['F1_avg_SIG_STR_landed'],
        'R_avg_TD_landed':        d['F1_avg_TD_landed'],
        'R_current_win_streak':   d['F1_current_win_streak'],
        'R_current_lose_streak':  d['F1_current_lose_streak'],
        'R_longest_win_streak':   d['F1_longest_win_streak'],
        'R_avg_SIG_STR_pct':      d['F1_avg_SIG_STR_pct'],
        'R_avg_SUB_ATT':          d['F1_avg_SUB_ATT'],
        'R_avg_TD_pct':           d['F1_avg_TD_pct'],
        'R_Reach_cms':            d['F1_Reach_cms'],
        'R_total_title_bouts':    d['F1_total_title_bouts'],

        'B_wins':                 d['F2_wins'],
        'B_losses':               d['F2_losses'],
        'B_Height_cms':           d['F2_Height_cms'],
        'B_age':                  d['F2_age'],
        'B_avg_SIG_STR_landed':   d['F2_avg_SIG_STR_landed'],
        'B_avg_TD_landed':        d['F2_avg_TD_landed'],
        'B_current_win_streak':   d['F2_current_win_streak'],
        'B_current_lose_streak':  d['F2_current_lose_streak'],
        'B_longest_win_streak':   d['F2_longest_win_streak'],
        'B_avg_SIG_STR_pct':      d['F2_avg_SIG_STR_pct'],
        'B_avg_SUB_ATT':          d['F2_avg_SUB_ATT'],
        'B_avg_TD_pct':           d['F2_avg_TD_pct'],
        'B_Reach_cms':            d['F2_Reach_cms'],
        'B_total_title_bouts':    d['F2_total_title_bouts'],

        # ── Diffs ─────────────────────────────────────────────────────────────
        'win_dif':              d['F1_wins']               - d['F2_wins'],
        'loss_dif':             d['F1_losses']             - d['F2_losses'],
        'win_streak_dif':       d['F1_current_win_streak'] - d['F2_current_win_streak'],
        'lose_streak_dif':      d['F1_current_lose_streak']- d['F2_current_lose_streak'],
        'height_dif':           d['F1_Height_cms']         - d['F2_Height_cms'],
        'reach_dif':            d['F1_Reach_cms']          - d['F2_Reach_cms'],
        'age_dif':              d['F1_age']                - d['F2_age'],
        'sig_str_dif':          d['F1_avg_SIG_STR_landed'] - d['F2_avg_SIG_STR_landed'],
        'avg_td_dif':           d['F1_avg_TD_landed']      - d['F2_avg_TD_landed'],
        'ko_dif':               d['F1_win_by_KO_TKO']      - d['F2_win_by_KO_TKO'],
        'sub_dif':              d['F1_win_by_Submission']  - d['F2_win_by_Submission'],
        'total_title_bout_dif': d['F1_total_title_bouts']  - d['F2_total_title_bouts'],

        # ── Match meta ────────────────────────────────────────────────────────
        'weight_class_ord': weight_class_ord,
        'title_bout_bin':   title_bout_bin,

        # ── Stance ────────────────────────────────────────────────────────────
        'orth_clash':  orth_clash,
        'south_clash': south_clash,
        'R_southpaw':  R_southpaw,
        'B_southpaw':  B_southpaw,

        # ── Career ────────────────────────────────────────────────────────────
        'R_cum_fights':          d['F1_cum_fights'],
        'B_cum_fights':          d['F2_cum_fights'],
        'R_career_win_rate':     d['F1_career_win_rate'],
        'B_career_win_rate':     d['F2_career_win_rate'],
        'career_win_rate_dif':   d['F1_career_win_rate']   - d['F2_career_win_rate'],
        'R_last5_won':           d['F1_last5_won'],
        'B_last5_won':           d['F2_last5_won'],
        'last5_won_dif':         d['F1_last5_won']         - d['F2_last5_won'],
        'R_last5_finish_rate':   d['F1_last5_finish_rate'],
        'B_last5_finish_rate':   d['F2_last5_finish_rate'],
        'last5_finish_rate_dif': d['F1_last5_finish_rate'] - d['F2_last5_finish_rate'],
        'R_opp_quality':         d['F1_opp_quality'],
        'B_opp_quality':         d['F2_opp_quality'],
        'opp_quality_dif':       d['F1_opp_quality']       - d['F2_opp_quality'],
        'R_trend_score':         d['F1_trend_score'],
        'B_trend_score':         d['F2_trend_score'],
        'trend_score_dif':       d['F1_trend_score']       - d['F2_trend_score'],
        'R_ko_finish_rate':      d['F1_ko_finish_rate'],
        'B_ko_finish_rate':      d['F2_ko_finish_rate'],
        'ko_finish_rate_dif':    d['F1_ko_finish_rate']    - d['F2_ko_finish_rate'],
        'R_sub_finish_rate':     d['F1_sub_finish_rate'],
        'B_sub_finish_rate':     d['F2_sub_finish_rate'],
        'sub_finish_rate_dif':   d['F1_sub_finish_rate']   - d['F2_sub_finish_rate'],
        'R_last3_win_rate':      d['F1_last3_win_rate'],
        'B_last3_win_rate':      d['F2_last3_win_rate'],
        'last3_win_rate_dif':    d['F1_last3_win_rate']    - d['F2_last3_win_rate'],
        'R_last10_win_rate':     d['F1_last10_win_rate'],
        'B_last10_win_rate':     d['F2_last10_win_rate'],
        'last10_win_rate_dif':   d['F1_last10_win_rate']   - d['F2_last10_win_rate'],
        'R_age_x_exp':           R_age_x_exp,
        'B_age_x_exp':           B_age_x_exp,
        'age_x_exp_dif':         R_age_x_exp               - B_age_x_exp,

        # ── Layoff buckets ────────────────────────────────────────────────────
        'R_layoff_lt90':    R_lb['lt90'],    'R_layoff_90_180':  R_lb['90_180'],
        'R_layoff_180_365': R_lb['180_365'], 'R_layoff_gt365':   R_lb['gt365'],
        'B_layoff_lt90':    B_lb['lt90'],    'B_layoff_90_180':  B_lb['90_180'],
        'B_layoff_180_365': B_lb['180_365'], 'B_layoff_gt365':   B_lb['gt365'],

        # ── Style ─────────────────────────────────────────────────────────────
        'R_SLpM':    d['F1_SLpM'],  'B_SLpM':    d['F2_SLpM'],
        'R_SApM':    d['F1_SApM'],  'B_SApM':    d['F2_SApM'],
        'R_Str_Acc': d['F1_Str_Acc'],'B_Str_Acc': d['F2_Str_Acc'],
        'R_Str_Def': d['F1_Str_Def'],'B_Str_Def': d['F2_Str_Def'],
        'R_TD_Avg':  d['F1_TD_Avg'], 'B_TD_Avg':  d['F2_TD_Avg'],
        'R_TD_Acc':  d['F1_TD_Acc'], 'B_TD_Acc':  d['F2_TD_Acc'],
        'R_TD_Def':  d['F1_TD_Def'], 'B_TD_Def':  d['F2_TD_Def'],
        'R_Sub_Avg': d['F1_Sub_Avg'],'B_Sub_Avg': d['F2_Sub_Avg'],
        'SLpM_dif':    d['F1_SLpM']    - d['F2_SLpM'],
        'SApM_dif':    d['F1_SApM']    - d['F2_SApM'],
        'Str_Def_dif': d['F1_Str_Def'] - d['F2_Str_Def'],
        'TD_Def_dif':  d['F1_TD_Def']  - d['F2_TD_Def'],
        'Sub_Avg_dif': d['F1_Sub_Avg'] - d['F2_Sub_Avg'],
        'TD_Avg_dif':  d['F1_TD_Avg']  - d['F2_TD_Avg'],

        # ── Elo features ──────────────────────────────────────────────────────
        'R_elo':         d['F1_elo'],
        'B_elo':         d['F2_elo'],
        'elo_dif':       d['F1_elo']       - d['F2_elo'],
        'R_elo_trend':   d['F1_elo_trend'],
        'B_elo_trend':   d['F2_elo_trend'],
        'elo_trend_dif': d['F1_elo_trend'] - d['F2_elo_trend'],

        # ── V2 interaction features (computable from existing inputs) ─────────
        # age × layoff: captures whether an older fighter took a long break
        'R_age_x_layoff':  d['F1_age'] * min(d['F1_layoff_days'], 730),
        'B_age_x_layoff':  d['F2_age'] * min(d['F2_layoff_days'], 730),
        'age_x_layoff_dif': (d['F1_age'] * min(d['F1_layoff_days'], 730)
                             - d['F2_age'] * min(d['F2_layoff_days'], 730)),
        # finish danger = offensive KO + sub threat
        'R_finish_danger': d['F1_ko_finish_rate'] + d['F1_sub_finish_rate'],
        'B_finish_danger': d['F2_ko_finish_rate'] + d['F2_sub_finish_rate'],
        # finish_danger_mismatch: needs got_finished_rate (chin) — default 0.5 (neutral)
        # R_finish_resistance = 1 - R_got_finished_rate; B same
        'finish_danger_mismatch': (
            (d['F1_ko_finish_rate'] + d['F1_sub_finish_rate']) * 0.5 -
            (d['F2_ko_finish_rate'] + d['F2_sub_finish_rate']) * 0.5
        ),
        'R_got_finished_rate': 0.5,
        'B_got_finished_rate': 0.5,
        # QA stats (require fighter-name career elo lookup — default to neutral)
        # The LR component (70%) has regularized these weights; XGB falls back on
        # the other 109 features when qa_* are 0/0.5.
        'R_qa_win_rate':    d['F1_career_win_rate'],
        'B_qa_win_rate':    d['F2_career_win_rate'],
        'qa_win_rate_dif':  d['F1_career_win_rate'] - d['F2_career_win_rate'],
        'R_qa_finish_rate': d['F1_last5_finish_rate'],
        'B_qa_finish_rate': d['F2_last5_finish_rate'],
        'qa_finish_rate_dif': d['F1_last5_finish_rate'] - d['F2_last5_finish_rate'],
        'R_qa_SLpM':  0.0,
        'B_qa_SLpM':  0.0,
        'qa_SLpM_dif': 0.0,
        'R_qa_SApM':  0.0,
        'B_qa_SApM':  0.0,
        'qa_SApM_dif': 0.0,
    }

    # ── Women's prediction path ───────────────────────────────────────────────
    if _is_womens and WOMENS_MODEL_AVAILABLE:
        df_w = pd.DataFrame([data])
        for col in womens_features:
            if col not in df_w.columns:
                df_w[col] = 0
        df_w = df_w[womens_features]

        x_w = df_w.values
        p_lr_w  = womens_model_lr.predict_proba(x_w)[0]
        p_xgb_w = womens_model_xgb.predict_proba(x_w)[0]
        prob_w   = LR_WEIGHT * p_lr_w + XGB_WEIGHT * p_xgb_w
        f1_prob  = float(prob_w[1])
        f2_prob  = float(prob_w[0])
        prediction = 'Fighter 1' if f1_prob > 0.5 else 'Fighter 2'
        confidence = round(max(f1_prob, f2_prob) * 100, 1)

        resp = {
            "prediction":        prediction,
            "confidence":        confidence,
            "f1_probability":    round(f1_prob * 100, 1),
            "f2_probability":    round(f2_prob * 100, 1),
            "upset_alert":       prediction == 'Fighter 2' and confidence > 55,
            "m1_prediction":     prediction,
            "m1_f1_probability": round(f1_prob * 100, 1),
            "m1_f2_probability": round(f2_prob * 100, 1),
            "m1_confidence":     confidence,
            "primary_prediction": prediction,
            "models_agree":      None,
            "upset_watch":       None,
            "model_scope":       "womens",
            "women_model_note":  f"baseline v1 · {_womens_acc*100:.2f}% accuracy",
        }

        # Women's M2A (full-feature odds blend) — inline since it needs fighter stats
        if WOMENS_M2A_AVAILABLE and d.get('f1_odds') is not None and d.get('f2_odds') is not None:
            f1_impl = _ml_to_impl_prob(float(d['f1_odds']))
            b_impl  = _ml_to_impl_prob(float(d['f2_odds']))
            odds_data = {
                **data,
                'R_impl_prob':   f1_impl,
                'B_impl_prob':   b_impl,
                'impl_prob_dif': f1_impl - b_impl,
                'market_fav_R':  1 if f1_impl > 0.5 else 0,
                'R_dec_impl':    _ml_to_impl_prob(float(d.get('f1_dec_odds') or 0)),
                'B_dec_impl':    _ml_to_impl_prob(float(d.get('f2_dec_odds') or 0)),
                'R_ko_impl':     _ml_to_impl_prob(float(d.get('f1_ko_odds') or 0)),
                'B_ko_impl':     _ml_to_impl_prob(float(d.get('f2_ko_odds') or 0)),
                'R_sub_impl':    _ml_to_impl_prob(float(d.get('f1_sub_odds') or 0)),
                'B_sub_impl':    _ml_to_impl_prob(float(d.get('f2_sub_odds') or 0)),
            }
            df_m2a = pd.DataFrame([odds_data])
            for col in womens_m2a_features:
                if col not in df_m2a.columns:
                    df_m2a[col] = 0
            df_m2a = df_m2a[womens_m2a_features]

            p_lr_m2  = womens_m2a_lr.predict_proba(df_m2a.values)[0]
            p_xgb_m2 = womens_m2a_xgb.predict_proba(df_m2a.values)[0]
            p_m2     = 0.5 * p_lr_m2 + 0.5 * p_xgb_m2
            m2a_f1   = float(p_m2[1])
            m2a_pred = 'Fighter 1' if m2a_f1 > 0.5 else 'Fighter 2'
            m2a_conf = round(max(m2a_f1, 1 - m2a_f1) * 100, 1)
            models_agree = (f1_prob > 0.5) == (m2a_f1 > 0.5)
            resp.update({
                "m2a_prediction":     m2a_pred,
                "m2a_f1_probability": round(m2a_f1 * 100, 1),
                "m2a_f2_probability": round((1 - m2a_f1) * 100, 1),
                "m2a_confidence":     m2a_conf,
                "primary_prediction": m2a_pred,
                "models_agree":       models_agree,
                "upset_watch":        not models_agree,
            })

        return resp

    # ── Men's prediction path (original logic) ────────────────────────────────
    df_input = pd.DataFrame([data])
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]

    x = df_input.values
    prob_lr  = model_lr.predict_proba(x)[0]
    prob_xgb = model_xgb.predict_proba(x)[0]
    prob     = LR_WEIGHT * prob_lr + XGB_WEIGHT * prob_xgb
    f1_prob  = float(prob[1])
    f2_prob  = float(prob[0])
    prediction  = 'Fighter 1' if f1_prob > 0.5 else 'Fighter 2'
    confidence  = round(max(f1_prob, f2_prob) * 100, 1)

    resp = {
        "prediction":      prediction,
        "confidence":      confidence,
        "f1_probability":  round(f1_prob * 100, 1),
        "f2_probability":  round(f2_prob * 100, 1),
        "upset_alert":     prediction == 'Fighter 2' and confidence > 55,
        "m1_prediction":   prediction,
        "m1_f1_probability": round(f1_prob * 100, 1),
        "m1_f2_probability": round(f2_prob * 100, 1),
        "m1_confidence":   confidence,
        "primary_prediction": prediction,
        "models_agree":    None,
        "upset_watch":     None,
        "model_scope":     "mens",
    }

    if _M2_READY and d.get('f1_odds') is not None and d.get('f2_odds') is not None:
        m2a_in = Model2Input(
            model1_prob=f1_prob,
            f1_odds=float(d['f1_odds']),
            f2_odds=float(d['f2_odds']),
            f1_dec_odds=float(d.get('f1_dec_odds') or 0),
            f2_dec_odds=float(d.get('f2_dec_odds') or 0),
            f1_ko_odds=float(d.get('f1_ko_odds') or 0),
            f2_ko_odds=float(d.get('f2_ko_odds') or 0),
            f1_sub_odds=float(d.get('f1_sub_odds') or 0),
            f2_sub_odds=float(d.get('f2_sub_odds') or 0),
            f1_ko_finish_rate=float(d.get('F1_ko_finish_rate') or 0),
            f2_ko_finish_rate=float(d.get('F2_ko_finish_rate') or 0),
            f1_sub_finish_rate=float(d.get('F1_sub_finish_rate') or 0),
            f2_sub_finish_rate=float(d.get('F2_sub_finish_rate') or 0),
            str_def_dif=float(d.get('F1_Str_Def', 0) - d.get('F2_Str_Def', 0)),
            weight_class=d.get('weight_class', 'Welterweight'),
            no_of_rounds=int(d.get('no_of_rounds', 3)),
        )
        m2a_r = model2a_predict(m2a_in)
        m2a_f1_pct = m2a_r['m2_prob_f1']
        m2a_f2_pct = m2a_r['m2_prob_f2']
        m2a_pred   = 'Fighter 1' if m2a_f1_pct > 50 else 'Fighter 2'
        m2a_conf   = round(max(m2a_f1_pct, m2a_f2_pct), 1)
        models_agree = (f1_prob > 0.5) == (m2a_f1_pct > 50)
        resp.update({
            "m2a_prediction":    m2a_pred,
            "m2a_f1_probability": m2a_f1_pct,
            "m2a_f2_probability": m2a_f2_pct,
            "m2a_confidence":    m2a_conf,
            "primary_prediction": m2a_pred,
            "models_agree":      models_agree,
            "upset_watch":       not models_agree,
        })

    return resp


# ─────────────────────────────────────────────────────────────────────────────
# /model2  — odds-aware prediction + ROI optimizer sizing
# ─────────────────────────────────────────────────────────────────────────────
class Model2Input(BaseModel):
    model1_prob:       float           # Model 1 P(F1 wins), 0–1 scale
    f1_odds:           float           # American moneyline (e.g. -150 or +130)
    f2_odds:           float
    f1_dec_odds:       float = 0       # 0 = not available
    f2_dec_odds:       float = 0
    f1_ko_odds:        float = 0
    f2_ko_odds:        float = 0
    f1_sub_odds:       float = 0
    f2_sub_odds:       float = 0
    f1_ufc_wins:       float = 0       # for debutant cap
    f2_ufc_wins:       float = 0
    # Optional fighter stats for new 42-feature model
    f1_ko_finish_rate: float = 0.0
    f2_ko_finish_rate: float = 0.0
    f1_sub_finish_rate: float = 0.0
    f2_sub_finish_rate: float = 0.0
    str_def_dif:       float = 0.0     # R_Str_Def - B_Str_Def (positive = F1 better)
    weight_class:      str  = 'Welterweight'
    no_of_rounds:      int  = 3


_WC_ORDER_M2 = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}


def _implied(odds: float) -> float:
    """American odds → raw implied probability. Returns 0.5 for missing (0)."""
    if odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def _decimal(odds: float):
    """American odds → decimal odds. Returns None for missing (0)."""
    if odds == 0:
        return None
    if odds < 0:
        return 1 + 100 / abs(odds)
    return 1 + odds / 100


@app.post("/model2a")
@app.post("/model2")
def model2a_predict(data: Model2Input):
    from fastapi import HTTPException
    if not _M2_READY:
        raise HTTPException(status_code=503, detail="Model 2 not loaded")

    # ── No-vig moneyline ──────────────────────────────────────────────────────
    f1_raw   = _implied(data.f1_odds)
    f2_raw   = _implied(data.f2_odds)
    total    = f1_raw + f2_raw
    f1_novig = f1_raw / total
    f2_novig = f2_raw / total
    vig      = total - 1.0
    ml_gap   = data.model1_prob - f1_novig

    # ── Method odds (use defaults when unavailable) ───────────────────────────
    f1_dec = _implied(data.f1_dec_odds) if data.f1_dec_odds != 0 else 0.5
    f2_dec = _implied(data.f2_dec_odds) if data.f2_dec_odds != 0 else 0.5
    f1_ko  = _implied(data.f1_ko_odds)  if data.f1_ko_odds  != 0 else 0.3
    f2_ko  = _implied(data.f2_ko_odds)  if data.f2_ko_odds  != 0 else 0.3
    f1_sub = _implied(data.f1_sub_odds) if data.f1_sub_odds != 0 else 0.2
    f2_sub = _implied(data.f2_sub_odds) if data.f2_sub_odds != 0 else 0.2

    # ── Base 23 features ─────────────────────────────────────────────────────
    dec_total   = f1_dec + f2_dec
    finish_prob = 1.0 - (dec_total / 2.0) if dec_total > 0 else 0.5
    f1_fin      = f1_ko + f1_sub
    f2_fin      = f2_ko + f2_sub
    model_conf  = abs(data.model1_prob - 0.5)
    vegas_conf  = abs(f1_novig - 0.5)

    base_features = {
        'model1_prob':        data.model1_prob,
        'f1_ml_novig':        f1_novig,
        'f2_ml_novig':        f2_novig,
        'ml_gap':             ml_gap,
        'vig':                vig,
        'f1_dec_implied':     f1_dec,
        'f2_dec_implied':     f2_dec,
        'dec_implied_dif':    f1_dec - f2_dec,
        'f1_ko_implied':      f1_ko,
        'f2_ko_implied':      f2_ko,
        'ko_implied_dif':     f1_ko - f2_ko,
        'f1_sub_implied':     f1_sub,
        'f2_sub_implied':     f2_sub,
        'sub_implied_dif':    f1_sub - f2_sub,
        'finish_prob':        finish_prob,
        'f1_finish_prob':     f1_fin,
        'f2_finish_prob':     f2_fin,
        'finish_advantage':   f1_fin - f2_fin,
        'abs_gap':            abs(ml_gap),
        'vegas_confidence':   vegas_conf,
        'model_confidence':   model_conf,
        'model_agrees_vegas': 1 if (data.model1_prob > 0.5) == (f1_novig > 0.5) else 0,
        'gap_x_confidence':   ml_gap * vegas_conf,
    }

    # ── Step 1: underdog/favorite profile features ────────────────────────────
    # f1_is_fav, odds_strength, tier_hist_win_rate computable from odds.
    # Per-fighter history features (fav/dog win rates) default to neutral (0.5/0)
    # — these had low correlation (r≤0.07) and ElasticNet will have shrunk their weights.
    f1_is_fav  = 1 if data.f1_odds < 0 else 0
    odds_str   = abs(f1_novig - 0.5)

    if f1_novig < 0.30:      tier = 0
    elif f1_novig < 0.45:    tier = 1
    elif f1_novig < 0.55:    tier = 2
    elif f1_novig < 0.70:    tier = 3
    else:                    tier = 4
    tier_wr_map = _m2_tier_stats.get('tier_win_rates', {})
    tier_hist_wr = float(tier_wr_map.get(str(tier), 0.5))

    step1_features = {
        'f1_is_fav':         f1_is_fav,
        'f1_hist_fav_wr':    0.5,   # fighter-specific; default neutral
        'f1_hist_dog_wr':    0.5,   # fighter-specific; default neutral
        'f1_fav_bouts_log':  0.0,
        'f1_dog_bouts_log':  0.0,
        'odds_strength':     odds_str,
        'tier_hist_win_rate': tier_hist_wr,
    }

    # ── Step 2: method odds × fighter style interactions ─────────────────────
    ko_style_edge   = f1_ko * data.f1_ko_finish_rate - f2_ko * data.f2_ko_finish_rate
    sub_style_edge  = f1_sub * data.f1_sub_finish_rate - f2_sub * data.f2_sub_finish_rate
    fin_x_conf      = finish_prob * model_conf
    dec_x_str_def   = ((f1_dec + f2_dec) / 2.0) * abs(data.str_def_dif)

    step2_features = {
        'ko_style_edge':        ko_style_edge,
        'sub_style_edge':       sub_style_edge,
        'finish_x_model_conf':  fin_x_conf,
        'dec_x_str_def':        dec_x_str_def,
        'combined_ko_implied':  f1_ko + f2_ko,
        'combined_sub_implied': f1_sub + f2_sub,
        'ko_method_gap':        abs(f1_ko - f2_ko),
        'sub_method_gap':       abs(f1_sub - f2_sub),
    }

    # ── Step 3: weight class and fight context ────────────────────────────────
    wc_ord   = _WC_ORDER_M2.get(data.weight_class, 8)
    wc_norm  = wc_ord / 11.0
    is_5r    = 1 if data.no_of_rounds >= 5 else 0
    m1_train_acc = float(_m2_tier_stats.get('m1_train_acc', 0.6405))
    m1_wc_acc_map = _m2_tier_stats.get('m1_wc_acc', {})
    m1_wc_acc = float(m1_wc_acc_map.get(str(wc_ord), m1_train_acc))
    m1_wc_bias = m1_wc_acc - m1_train_acc

    step3_features = {
        'wc_norm':       wc_norm,
        'is_5r':         is_5r,
        'm1_wc_bias':    m1_wc_bias,
        'five_r_x_conf': is_5r * model_conf,
    }

    # ── Build 42-feature row and run 50/50 blend ──────────────────────────────
    features_dict = {**base_features, **step1_features, **step2_features, **step3_features}
    df_in = pd.DataFrame([features_dict])[model2_features]

    lr_prob_f1  = float(model2.predict_proba(df_in)[0][1])
    xgb_prob_f1 = float(model2_xgb.predict_proba(df_in)[0][1])
    m2_prob_f1  = M2_LR_WEIGHT * lr_prob_f1 + M2_XGB_WEIGHT * xgb_prob_f1
    m2_prob_f2  = 1.0 - m2_prob_f1

    # ── Gap between Model 2 and Vegas ─────────────────────────────────────────
    final_gap = m2_prob_f1 - f1_novig
    pick      = 'f1' if final_gap > 0 else 'f2'
    pick_prob = m2_prob_f1 if pick == 'f1' else m2_prob_f2
    pick_odds = data.f1_odds if pick == 'f1' else data.f2_odds

    # ── Kelly sizing ──────────────────────────────────────────────────────────
    is_value = abs(final_gap) >= GAP_THRESHOLD
    bet_size = 0

    if is_value:
        dec = _decimal(pick_odds)
        if dec is not None and dec > 1:
            b         = dec - 1.0
            kelly_pct = max(0.0, (pick_prob * b - (1.0 - pick_prob)) / b)
            eff_kelly = kelly_pct * KELLY_FRACTION
            if (pick == 'f1' and data.f1_ufc_wins == 0) or \
               (pick == 'f2' and data.f2_ufc_wins == 0):
                eff_kelly *= 0.5
            bet_size = min(MAX_BET, round(eff_kelly * BANKROLL))

    return {
        "m2_prob_f1":     round(m2_prob_f1 * 100, 1),
        "m2_prob_f2":     round(m2_prob_f2 * 100, 1),
        "f1_novig":       round(f1_novig * 100, 1),
        "f2_novig":       round(f2_novig * 100, 1),
        "ml_gap":         round(ml_gap * 100, 1),
        "final_gap":      round(final_gap * 100, 1),
        "pick":           pick,
        "pick_prob":      round(pick_prob * 100, 1),
        "pick_odds":      pick_odds,
        "is_value":       is_value,
        "bet_size":       bet_size,
        "threshold_used": int(GAP_THRESHOLD * 100),
        "kelly_fraction": f"1/{int(1 / KELLY_FRACTION)}",
        "m2_blend":       "50% LR + 50% XGB",
    }


# ─────────────────────────────────────────────────────────────────────────────
# /model2b  — gap zone calibration (The Bettor)
# Accepts same inputs as /model2a, recomputes M2A internally, runs M2B ensemble.
# ─────────────────────────────────────────────────────────────────────────────
_ZONE_LABELS = {0:'Lock',1:'Strong',2:'Lean',3:'Watch',4:'Value',5:'Strong Value',6:'Max Value'}

def _gap_zone(gap_size: float) -> int:
    if gap_size < 0.01:  return 0
    elif gap_size < 0.02: return 1
    elif gap_size < 0.03: return 2
    elif gap_size < 0.05: return 3
    elif gap_size < 0.08: return 4
    elif gap_size < 0.10: return 5
    else:                 return 6

def _m2b_conf_label(prob: float) -> str:
    if prob >= 0.75:   return 'LOCK'
    elif prob >= 0.65: return 'HIGH'
    elif prob >= 0.55: return 'MEDIUM'
    else:              return 'LOW'

def _m2b_action(label: str, triple_agree: bool) -> str:
    if label == 'LOCK' and triple_agree:  return 'Strong bet — all three models converge'
    elif label == 'LOCK':                 return 'Strong bet — M1+M2A converge'
    elif label == 'HIGH' and triple_agree: return 'Bet — high confidence, Vegas agrees'
    elif label == 'HIGH':                 return 'Bet — M1+M2A agree'
    elif label == 'MEDIUM':               return 'Lean — monitor line movement'
    else:                                 return 'Avoid — low confidence'


_WC_ORDER_M2B = {
    "Flyweight":          0, "Strawweight":       0,
    "Bantamweight":       1, "Featherweight":     2,
    "Lightweight":        3, "Welterweight":      4,
    "Middleweight":       5, "Light Heavyweight": 6,
    "Heavyweight":        7, "Catch Weight":      4,
}

def _odds_tier_m2b(ml: float) -> int:
    """Moneyline → 0-6 ordinal tier (matches V3 training: hfav=0 … hdog=6)."""
    if ml == 0:    return 3
    if ml < -300:  return 0
    if ml < -150:  return 1
    if ml < -110:  return 2
    if ml <=  110: return 3
    if ml <=  200: return 4
    if ml <=  400: return 5
    return 6

@app.post("/model2b")
def model2b_predict(data: Model2Input):
    from fastapi import HTTPException
    if not _M2B_READY:
        raise HTTPException(status_code=503, detail="Model 2B not loaded")
    if not _M2_READY:
        raise HTTPException(status_code=503, detail="Model 2A not loaded (required for 2B)")

    # Run M2A to get m2a_prob_f1
    m2a_r    = model2a_predict(data)
    m2a_prob = m2a_r['m2_prob_f1'] / 100.0
    m1_prob  = data.model1_prob

    # No-vig
    f1_raw   = _implied(data.f1_odds)
    f2_raw   = _implied(data.f2_odds)
    total    = f1_raw + f2_raw
    f1_novig = f1_raw / total
    f2_novig = f2_raw / total

    m2a_picks_f1 = m2a_prob > 0.5
    m1_picks_f1  = m1_prob  > 0.5
    vegas_fav_f1 = f1_novig > 0.5

    pick_prob_val  = m2a_prob if m2a_picks_f1 else 1.0 - m2a_prob
    pick_novig_val = f1_novig if m2a_picks_f1 else f2_novig
    closing_odds   = data.f1_odds if m2a_picks_f1 else data.f2_odds

    gap       = pick_prob_val - pick_novig_val
    gap_size  = abs(gap)
    gap_dir   = 1.0 if gap >= 0 else -1.0
    gap_signed = gap_size * gap_dir
    zone      = _gap_zone(gap_size)

    m1_m2a_agree = int(m1_picks_f1 == m2a_picks_f1)
    vegas_agree  = int(m2a_picks_f1 == vegas_fav_f1)
    triple_agree = int(m1_m2a_agree == 1 and vegas_agree == 1)

    m1_conviction      = abs(m1_prob  - 0.5)
    m2a_conviction     = abs(m2a_prob - 0.5)
    conviction_product = m1_conviction * m2a_conviction
    conviction_gap     = abs(m1_prob - m2a_prob)

    # odds_tier: moneyline-based 0-6 (matches V3 training)
    odds_tier = _odds_tier_m2b(closing_odds)

    # weight_class_ord: 0-7 integer (matches V3 training WEIGHT_CLASS_ORD)
    wc_ord = _WC_ORDER_M2B.get(data.weight_class, 4)
    is_5r  = 1 if data.no_of_rounds >= 5 else 0

    # one-sided vig: implied(closing_odds) - pick_novig, clipped [0, 0.15]
    vig_1side = max(0.0, min(0.15, _implied(closing_odds) - pick_novig_val))

    # agreement type + ordinal encoding (matches V3 training)
    if m1_m2a_agree == 0:
        agreement_type    = 'SPLIT'
        agreement_encoded = 2
    elif vegas_agree == 0 and gap_dir >= 0:
        agreement_type    = 'CONFIRM_DOG'   # underdog pick, positive gap — highest value
        agreement_encoded = 3
    elif vegas_agree == 1 and gap_dir >= 0:
        agreement_type    = 'CONFIRM_FAV'   # favorite pick, positive gap
        agreement_encoded = 3
    else:
        agreement_type    = 'NO_EDGE'       # gap <= 0, no bet
        agreement_encoded = 0

    # is_m1_signal: hardcoded 0 — inverted in training data, display removed
    is_m1_signal = 0

    feat_row = np.array([[
        gap_size,                  # 0  gap_size
        float(zone),               # 1  gap_zone
        gap_dir,                   # 2  gap_direction
        gap_signed,                # 3  gap_signed
        m1_prob,                   # 4  m1_prob
        m2a_prob,                  # 5  m2a_prob
        m1_conviction,             # 6  m1_conviction
        m2a_conviction,            # 7  m2a_conviction
        float(m1_m2a_agree),       # 8  m1_m2a_agree
        float(vegas_agree),        # 9  vegas_agree
        float(triple_agree),       # 10 triple_agree
        conviction_product,        # 11 conviction_product
        conviction_gap,            # 12 conviction_gap
        float(odds_tier),          # 13 odds_tier
        float(wc_ord),             # 14 weight_class_ord
        float(is_5r),              # 15 is_5round
        vig_1side,                 # 16 vig
        float(closing_odds),       # 17 closing_odds
        float(is_m1_signal),       # 18 is_m1_signal
        float(agreement_encoded),  # 19 agreement_encoded
    ]])

    win_prob = float(_model2b_rf.predict_proba(feat_row)[0][1])

    # SPLIT floor: SPLIT fights actual WR 52.1%; floor prevents systematic underconfidence
    split_floor = _model2b_config.get('split_floor', 0.45)
    if agreement_type == 'SPLIT':
        win_prob = max(win_prob, split_floor)

    conf_label = _m2b_conf_label(win_prob)
    return {
        "win_probability":     round(win_prob * 100, 1),
        "gap_zone":            zone,
        "gap_zone_label":      _ZONE_LABELS[zone],
        "gap_size_pct":        round(gap_size * 100, 2),
        "gap_direction":       int(gap_dir),
        "agreement_type":      agreement_type,
        "m1_m2a_agree":        bool(m1_m2a_agree),
        "vegas_agree":         bool(vegas_agree),
        "triple_agree":        bool(triple_agree),
        "confidence_label":    conf_label,
        "recommended_action":  _m2b_action(conf_label, bool(triple_agree)),
        "model_type":          "RF + SPLIT floor",
        "split_floor_applied": agreement_type == 'SPLIT',
    }


# ─────────────────────────────────────────────────────────────────────────────
# /method  — Models 3A (Goes the Distance) + 3B (Winner+Method)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/method")
def predict_method(fight: FightInput):
    if not MODEL3_AVAILABLE:
        return {"error": "Method prediction models not loaded"}

    d = fight.dict()
    wc_name       = d['weight_class']
    is_womens     = is_womens_fight(wc_name)
    wc_ord        = WC_ORDER.get(wc_name, 8)
    is_5rnd       = 1 if d['no_of_rounds'] >= 5 else 0
    is_title      = 1 if d['title_bout'] else 0
    is_womens_flg = 1 if is_womens else 0
    low_conf_div  = wc_name in _M3_LOW_CONF_DIVISIONS

    f1_label = d.get('f1_name') or 'Fighter 1'
    f2_label = d.get('f2_name') or 'Fighter 2'

    r_cr = get_career_method_rates(d.get('f1_name') or '')
    b_cr = get_career_method_rates(d.get('f2_name') or '')

    # ── 3A feature dict (63 features) ────────────────────────────────────────
    feat3a = {
        'weight_class_ord': wc_ord,
        'is_5rnd':          is_5rnd,
        'is_title':         is_title,
        'is_womens':        is_womens_flg,

        'R_career_is_finish':        r_cr['career_is_finish'],
        'R_career_is_decision':      r_cr['career_is_decision'],
        'R_career_is_ko':            r_cr['career_is_ko'],
        'R_career_is_sub':           r_cr['career_is_sub'],
        'R_career_finish_delivered': r_cr['career_finish_delivered'],
        'R_career_finish_received':  r_cr['career_finish_received'],
        'R_career_dec_delivered':    r_cr['career_dec_delivered'],
        'R_career_dec_received':     r_cr['career_dec_received'],
        'R_career_n_fights':         r_cr['career_n_fights'],

        'B_career_is_finish':        b_cr['career_is_finish'],
        'B_career_is_decision':      b_cr['career_is_decision'],
        'B_career_is_ko':            b_cr['career_is_ko'],
        'B_career_is_sub':           b_cr['career_is_sub'],
        'B_career_finish_delivered': b_cr['career_finish_delivered'],
        'B_career_finish_received':  b_cr['career_finish_received'],
        'B_career_dec_delivered':    b_cr['career_dec_delivered'],
        'B_career_dec_received':     b_cr['career_dec_received'],
        'B_career_n_fights':         b_cr['career_n_fights'],

        'combined_is_finish':        r_cr['career_is_finish']        + b_cr['career_is_finish'],
        'combined_is_decision':      r_cr['career_is_decision']      + b_cr['career_is_decision'],
        'combined_is_ko':            r_cr['career_is_ko']            + b_cr['career_is_ko'],
        'combined_is_sub':           r_cr['career_is_sub']           + b_cr['career_is_sub'],
        'combined_finish_delivered': r_cr['career_finish_delivered'] + b_cr['career_finish_delivered'],
        'combined_finish_received':  r_cr['career_finish_received']  + b_cr['career_finish_received'],
        'combined_dec_delivered':    r_cr['career_dec_delivered']    + b_cr['career_dec_delivered'],
        'combined_dec_received':     r_cr['career_dec_received']     + b_cr['career_dec_received'],

        'R_SLpM':    d['F1_SLpM'],    'R_SApM':    d['F1_SApM'],
        'R_Str_Def': d['F1_Str_Def'], 'R_TD_Avg':  d['F1_TD_Avg'],
        'R_TD_Def':  d['F1_TD_Def'],  'R_Sub_Avg': d['F1_Sub_Avg'],
        'B_SLpM':    d['F2_SLpM'],    'B_SApM':    d['F2_SApM'],
        'B_Str_Def': d['F2_Str_Def'], 'B_TD_Avg':  d['F2_TD_Avg'],
        'B_TD_Def':  d['F2_TD_Def'],  'B_Sub_Avg': d['F2_Sub_Avg'],

        'combined_SLpM':    d['F1_SLpM']    + d['F2_SLpM'],
        'combined_SApM':    d['F1_SApM']    + d['F2_SApM'],
        'combined_Str_Def': d['F1_Str_Def'] + d['F2_Str_Def'],
        'combined_TD_Avg':  d['F1_TD_Avg']  + d['F2_TD_Avg'],
        'combined_TD_Def':  d['F1_TD_Def']  + d['F2_TD_Def'],
        'combined_Sub_Avg': d['F1_Sub_Avg'] + d['F2_Sub_Avg'],

        'R_avg_SIG_STR_landed':    d['F1_avg_SIG_STR_landed'],
        'B_avg_SIG_STR_landed':    d['F2_avg_SIG_STR_landed'],
        'combined_sig_str_landed': d['F1_avg_SIG_STR_landed'] + d['F2_avg_SIG_STR_landed'],
        'R_avg_TD_landed':         d['F1_avg_TD_landed'],
        'B_avg_TD_landed':         d['F2_avg_TD_landed'],
        'combined_td_landed':      d['F1_avg_TD_landed']      + d['F2_avg_TD_landed'],
        'R_avg_SUB_ATT':           d['F1_avg_SUB_ATT'],
        'B_avg_SUB_ATT':           d['F2_avg_SUB_ATT'],
        'combined_sub_att':        d['F1_avg_SUB_ATT']        + d['F2_avg_SUB_ATT'],

        'reach_dif':       d['F1_Reach_cms']         - d['F2_Reach_cms'],
        'age_dif':         d['F1_age']                - d['F2_age'],
        'sig_str_dif':     d['F1_avg_SIG_STR_landed'] - d['F2_avg_SIG_STR_landed'],
        'avg_sub_att_dif': d['F1_avg_SUB_ATT']        - d['F2_avg_SUB_ATT'],
        'ko_dif':          d['F1_win_by_KO_TKO']       - d['F2_win_by_KO_TKO'],
        'sub_dif':         d['F1_win_by_Submission']   - d['F2_win_by_Submission'],
    }

    df3a = pd.DataFrame([feat3a])[model3a_features]
    p_3a_lr  = model3a_lr.predict_proba(df3a)[0]
    p_3a_xgb = model3a_xgb.predict_proba(df3a)[0]
    p_3a = 0.30 * p_3a_lr + 0.70 * p_3a_xgb
    goes_distance_prob = float(p_3a[1])
    finish_prob        = float(p_3a[0])

    # ── Run M1 to get m1_red_win_prob for 3B ─────────────────────────────────
    R_southpaw  = int(d['F1_is_southpaw'])
    B_southpaw  = int(d['F2_is_southpaw'])
    R_age_x_exp = d['F1_age'] * d['F1_cum_fights']
    B_age_x_exp = d['F2_age'] * d['F2_cum_fights']
    R_lb = _layoff_buckets(d['F1_layoff_days'])
    B_lb = _layoff_buckets(d['F2_layoff_days'])
    wc_ord_m1   = WC_ORDER_WOMENS.get(wc_name, 1) if is_womens else WC_ORDER.get(wc_name, 6)

    m1_data = {
        'R_wins': d['F1_wins'], 'R_losses': d['F1_losses'],
        'R_Height_cms': d['F1_Height_cms'], 'R_age': d['F1_age'],
        'R_avg_SIG_STR_landed': d['F1_avg_SIG_STR_landed'],
        'R_avg_TD_landed': d['F1_avg_TD_landed'],
        'R_current_win_streak': d['F1_current_win_streak'],
        'R_current_lose_streak': d['F1_current_lose_streak'],
        'R_longest_win_streak': d['F1_longest_win_streak'],
        'R_avg_SIG_STR_pct': d['F1_avg_SIG_STR_pct'],
        'R_avg_SUB_ATT': d['F1_avg_SUB_ATT'],
        'R_avg_TD_pct': d['F1_avg_TD_pct'],
        'R_Reach_cms': d['F1_Reach_cms'],
        'R_total_title_bouts': d['F1_total_title_bouts'],
        'B_wins': d['F2_wins'], 'B_losses': d['F2_losses'],
        'B_Height_cms': d['F2_Height_cms'], 'B_age': d['F2_age'],
        'B_avg_SIG_STR_landed': d['F2_avg_SIG_STR_landed'],
        'B_avg_TD_landed': d['F2_avg_TD_landed'],
        'B_current_win_streak': d['F2_current_win_streak'],
        'B_current_lose_streak': d['F2_current_lose_streak'],
        'B_longest_win_streak': d['F2_longest_win_streak'],
        'B_avg_SIG_STR_pct': d['F2_avg_SIG_STR_pct'],
        'B_avg_SUB_ATT': d['F2_avg_SUB_ATT'],
        'B_avg_TD_pct': d['F2_avg_TD_pct'],
        'B_Reach_cms': d['F2_Reach_cms'],
        'B_total_title_bouts': d['F2_total_title_bouts'],
        'win_dif':              d['F1_wins']                - d['F2_wins'],
        'loss_dif':             d['F1_losses']              - d['F2_losses'],
        'win_streak_dif':       d['F1_current_win_streak']  - d['F2_current_win_streak'],
        'lose_streak_dif':      d['F1_current_lose_streak'] - d['F2_current_lose_streak'],
        'height_dif':           d['F1_Height_cms']          - d['F2_Height_cms'],
        'reach_dif':            d['F1_Reach_cms']           - d['F2_Reach_cms'],
        'age_dif':              d['F1_age']                 - d['F2_age'],
        'sig_str_dif':          d['F1_avg_SIG_STR_landed']  - d['F2_avg_SIG_STR_landed'],
        'avg_td_dif':           d['F1_avg_TD_landed']       - d['F2_avg_TD_landed'],
        'ko_dif':               d['F1_win_by_KO_TKO']       - d['F2_win_by_KO_TKO'],
        'sub_dif':              d['F1_win_by_Submission']   - d['F2_win_by_Submission'],
        'total_title_bout_dif': d['F1_total_title_bouts']  - d['F2_total_title_bouts'],
        'weight_class_ord':     wc_ord_m1,
        'title_bout_bin':       1 if d['title_bout'] else 0,
        'orth_clash':  1 if (R_southpaw == 0 and B_southpaw == 0) else 0,
        'south_clash': 1 if (R_southpaw == 1 and B_southpaw == 1) else 0,
        'R_southpaw': R_southpaw, 'B_southpaw': B_southpaw,
        'R_cum_fights': d['F1_cum_fights'], 'B_cum_fights': d['F2_cum_fights'],
        'R_career_win_rate': d['F1_career_win_rate'],
        'B_career_win_rate': d['F2_career_win_rate'],
        'career_win_rate_dif': d['F1_career_win_rate']   - d['F2_career_win_rate'],
        'R_last5_won': d['F1_last5_won'], 'B_last5_won': d['F2_last5_won'],
        'last5_won_dif': d['F1_last5_won'] - d['F2_last5_won'],
        'R_last5_finish_rate': d['F1_last5_finish_rate'],
        'B_last5_finish_rate': d['F2_last5_finish_rate'],
        'last5_finish_rate_dif': d['F1_last5_finish_rate'] - d['F2_last5_finish_rate'],
        'R_opp_quality': d['F1_opp_quality'], 'B_opp_quality': d['F2_opp_quality'],
        'opp_quality_dif': d['F1_opp_quality']   - d['F2_opp_quality'],
        'R_trend_score': d['F1_trend_score'], 'B_trend_score': d['F2_trend_score'],
        'trend_score_dif': d['F1_trend_score'] - d['F2_trend_score'],
        'R_ko_finish_rate': d['F1_ko_finish_rate'],
        'B_ko_finish_rate': d['F2_ko_finish_rate'],
        'ko_finish_rate_dif': d['F1_ko_finish_rate'] - d['F2_ko_finish_rate'],
        'R_sub_finish_rate': d['F1_sub_finish_rate'],
        'B_sub_finish_rate': d['F2_sub_finish_rate'],
        'sub_finish_rate_dif': d['F1_sub_finish_rate'] - d['F2_sub_finish_rate'],
        'R_last3_win_rate': d['F1_last3_win_rate'],
        'B_last3_win_rate': d['F2_last3_win_rate'],
        'last3_win_rate_dif': d['F1_last3_win_rate'] - d['F2_last3_win_rate'],
        'R_last10_win_rate': d['F1_last10_win_rate'],
        'B_last10_win_rate': d['F2_last10_win_rate'],
        'last10_win_rate_dif': d['F1_last10_win_rate'] - d['F2_last10_win_rate'],
        'R_age_x_exp': R_age_x_exp, 'B_age_x_exp': B_age_x_exp,
        'age_x_exp_dif': R_age_x_exp - B_age_x_exp,
        'R_layoff_lt90': R_lb['lt90'], 'R_layoff_90_180': R_lb['90_180'],
        'R_layoff_180_365': R_lb['180_365'], 'R_layoff_gt365': R_lb['gt365'],
        'B_layoff_lt90': B_lb['lt90'], 'B_layoff_90_180': B_lb['90_180'],
        'B_layoff_180_365': B_lb['180_365'], 'B_layoff_gt365': B_lb['gt365'],
        'R_SLpM': d['F1_SLpM'],    'B_SLpM': d['F2_SLpM'],
        'R_SApM': d['F1_SApM'],    'B_SApM': d['F2_SApM'],
        'R_Str_Acc': d['F1_Str_Acc'], 'B_Str_Acc': d['F2_Str_Acc'],
        'R_Str_Def': d['F1_Str_Def'], 'B_Str_Def': d['F2_Str_Def'],
        'R_TD_Avg': d['F1_TD_Avg'],   'B_TD_Avg': d['F2_TD_Avg'],
        'R_TD_Acc': d['F1_TD_Acc'],   'B_TD_Acc': d['F2_TD_Acc'],
        'R_TD_Def': d['F1_TD_Def'],   'B_TD_Def': d['F2_TD_Def'],
        'R_Sub_Avg': d['F1_Sub_Avg'], 'B_Sub_Avg': d['F2_Sub_Avg'],
        'SLpM_dif':    d['F1_SLpM']    - d['F2_SLpM'],
        'SApM_dif':    d['F1_SApM']    - d['F2_SApM'],
        'Str_Def_dif': d['F1_Str_Def'] - d['F2_Str_Def'],
        'TD_Def_dif':  d['F1_TD_Def']  - d['F2_TD_Def'],
        'Sub_Avg_dif': d['F1_Sub_Avg'] - d['F2_Sub_Avg'],
        'TD_Avg_dif':  d['F1_TD_Avg']  - d['F2_TD_Avg'],
        'R_elo': d['F1_elo'], 'B_elo': d['F2_elo'],
        'elo_dif': d['F1_elo'] - d['F2_elo'],
        'R_elo_trend': d['F1_elo_trend'], 'B_elo_trend': d['F2_elo_trend'],
        'elo_trend_dif': d['F1_elo_trend'] - d['F2_elo_trend'],
        'R_age_x_layoff':  d['F1_age'] * min(d['F1_layoff_days'], 730),
        'B_age_x_layoff':  d['F2_age'] * min(d['F2_layoff_days'], 730),
        'age_x_layoff_dif': (d['F1_age'] * min(d['F1_layoff_days'], 730)
                              - d['F2_age'] * min(d['F2_layoff_days'], 730)),
        'R_finish_danger': d['F1_ko_finish_rate'] + d['F1_sub_finish_rate'],
        'B_finish_danger': d['F2_ko_finish_rate'] + d['F2_sub_finish_rate'],
        'finish_danger_mismatch': (
            (d['F1_ko_finish_rate'] + d['F1_sub_finish_rate']) * 0.5 -
            (d['F2_ko_finish_rate'] + d['F2_sub_finish_rate']) * 0.5
        ),
        'R_got_finished_rate': 0.5, 'B_got_finished_rate': 0.5,
        'R_qa_win_rate': d['F1_career_win_rate'],
        'B_qa_win_rate': d['F2_career_win_rate'],
        'qa_win_rate_dif': d['F1_career_win_rate'] - d['F2_career_win_rate'],
        'R_qa_finish_rate': d['F1_last5_finish_rate'],
        'B_qa_finish_rate': d['F2_last5_finish_rate'],
        'qa_finish_rate_dif': d['F1_last5_finish_rate'] - d['F2_last5_finish_rate'],
        'R_qa_SLpM': 0.0, 'B_qa_SLpM': 0.0, 'qa_SLpM_dif': 0.0,
        'R_qa_SApM': 0.0, 'B_qa_SApM': 0.0, 'qa_SApM_dif': 0.0,
    }

    df_m1 = pd.DataFrame([m1_data])
    for col in feature_columns:
        if col not in df_m1.columns:
            df_m1[col] = 0
    df_m1 = df_m1[feature_columns]
    p_lr_m1  = model_lr.predict_proba(df_m1)[0]
    p_xgb_m1 = model_xgb.predict_proba(df_m1)[0]
    m1_prob       = float(LR_WEIGHT * p_lr_m1[1] + XGB_WEIGHT * p_xgb_m1[1])
    m1_confidence = abs(m1_prob - 0.5) * 2

    # ── 3B feature dict (102 features) ───────────────────────────────────────
    r_wins = max(1, d['F1_wins'])
    b_wins = max(1, d['F2_wins'])
    R_ko_win_rate  = d['F1_win_by_KO_TKO']      / r_wins
    R_sub_win_rate = d['F1_win_by_Submission']   / r_wins
    R_dec_win_rate = (d['F1_win_by_Decision_Unanimous'] + d['F1_win_by_Decision_Split']
                      + d['F1_win_by_Decision_Majority']) / r_wins
    B_ko_win_rate  = d['F2_win_by_KO_TKO']      / b_wins
    B_sub_win_rate = d['F2_win_by_Submission']   / b_wins
    B_dec_win_rate = (d['F2_win_by_Decision_Unanimous'] + d['F2_win_by_Decision_Split']
                      + d['F2_win_by_Decision_Majority']) / b_wins

    feat3b = {
        **feat3a,
        'R_elo': d['F1_elo'],       'B_elo': d['F2_elo'],
        'elo_dif': d['F1_elo']    - d['F2_elo'],
        'R_elo_trend': d['F1_elo_trend'], 'B_elo_trend': d['F2_elo_trend'],
        'elo_trend_dif': d['F1_elo_trend'] - d['F2_elo_trend'],
        'R_career_win_rate': d['F1_career_win_rate'],
        'B_career_win_rate': d['F2_career_win_rate'],
        'career_win_rate_dif': d['F1_career_win_rate'] - d['F2_career_win_rate'],
        'R_ko_win_rate':  R_ko_win_rate,   'B_ko_win_rate':  B_ko_win_rate,
        'ko_win_rate_dif': R_ko_win_rate  - B_ko_win_rate,
        'R_sub_win_rate': R_sub_win_rate,  'B_sub_win_rate': B_sub_win_rate,
        'sub_win_rate_dif': R_sub_win_rate - B_sub_win_rate,
        'R_dec_win_rate': R_dec_win_rate,  'B_dec_win_rate': B_dec_win_rate,
        'dec_win_rate_dif': R_dec_win_rate - B_dec_win_rate,
        'SLpM_dif':    d['F1_SLpM']    - d['F2_SLpM'],
        'SApM_dif':    d['F1_SApM']    - d['F2_SApM'],
        'Str_Def_dif': d['F1_Str_Def'] - d['F2_Str_Def'],
        'TD_Avg_dif':  d['F1_TD_Avg']  - d['F2_TD_Avg'],
        'Sub_Avg_dif': d['F1_Sub_Avg'] - d['F2_Sub_Avg'],
        'R_avg_SIG_STR_pct': d['F1_avg_SIG_STR_pct'],
        'B_avg_SIG_STR_pct': d['F2_avg_SIG_STR_pct'],
        'sig_str_pct_dif':   d['F1_avg_SIG_STR_pct'] - d['F2_avg_SIG_STR_pct'],
        'R_avg_TD_pct': d['F1_avg_TD_pct'],
        'B_avg_TD_pct': d['F2_avg_TD_pct'],
        'td_pct_dif':   d['F1_avg_TD_pct'] - d['F2_avg_TD_pct'],
        'win_streak_dif':  d['F1_current_win_streak']  - d['F2_current_win_streak'],
        'lose_streak_dif': d['F1_current_lose_streak'] - d['F2_current_lose_streak'],
        'win_dif':   d['F1_wins']   - d['F2_wins'],
        'loss_dif':  d['F1_losses'] - d['F2_losses'],
        'avg_td_dif': d['F1_avg_TD_landed'] - d['F2_avg_TD_landed'],
        'total_round_dif':      0,
        'total_title_bout_dif': d['F1_total_title_bouts'] - d['F2_total_title_bouts'],
        'm1_red_win_prob':    m1_prob,
        'm1_red_win_prob_sq': m1_prob ** 2,
        'm1_confidence':      m1_confidence,
    }

    df3b = pd.DataFrame([feat3b])[model3b_features]
    p_3b_rf  = model3b_rf.predict_proba(df3b)[0]
    p_3b_xgb = model3b_xgb.predict_proba(df3b)[0]
    p_3b = 0.40 * p_3b_rf + 0.60 * p_3b_xgb

    outcome_labels = [
        f"{f1_label} by KO",
        f"{f1_label} by Sub",
        f"{f1_label} by Dec",
        f"{f2_label} by KO",
        f"{f2_label} by Sub",
        f"{f2_label} by Dec",
    ]

    most_probable_idx  = int(np.argmax(p_3b))
    most_probable_prob = float(p_3b[most_probable_idx])

    return {
        'goes_distance_prob':    round(goes_distance_prob, 4),
        'finish_prob':           round(finish_prob, 4),
        'most_probable_outcome': outcome_labels[most_probable_idx] if most_probable_prob >= 0.25 else None,
        'most_probable_prob':    round(most_probable_prob, 4),
        'all_outcomes': {
            'f1_ko':  round(float(p_3b[0]), 4),
            'f1_sub': round(float(p_3b[1]), 4),
            'f1_dec': round(float(p_3b[2]), 4),
            'f2_ko':  round(float(p_3b[3]), 4),
            'f2_sub': round(float(p_3b[4]), 4),
            'f2_dec': round(float(p_3b[5]), 4),
        },
        'low_confidence_division': low_conf_div,
    }


# ─────────────────────────────────────────────────────────────────────────────
# /odds  — fetch DraftKings lines, snapshot, return movement
# ─────────────────────────────────────────────────────────────────────────────
_ODDS_API_KEY       = 'ed5357f84b07c0850c6d112a61934725'
_ODDS_SNAPSHOT_FILE = '../data/odds_snapshots.json'

_CARD_FIGHTS = [
    ('Aljamain Sterling',    'Youssef Zalal'),
    ('Norma Dumont',         'Joselyne Edwards'),
    ('Rafa Garcia',          'Alexander Hernandez'),
    ('Davey Grant',          'Adrian Luna Martinetti'),
    ('Montel Jackson',       'Raoni Barcelos'),
    ('Marcus Buchecha',      'Ryan Spann'),
    ('Rodolfo Vieira',       'Eric McConico'),
    ('Jackson McVey',        'Sedriques Dumas'),
    ('Mayra Bueno Silva',    'Michelle Montague'),
    ('Jafel Filho',          'Cody Durden'),
    ('Francis Marshall',     'Lucas Brennan'),
    ('Max Griffin',          'Victor Valenzuela'),
    ('Talita Alencar',       'Julia Polastri'),
]


def _name_sim(a: str, b: str) -> float:
    a = re.sub(r"[^a-z ]", "", a.lower().strip())
    b = re.sub(r"[^a-z ]", "", b.lower().strip())
    return SequenceMatcher(None, a, b).ratio()


@app.get("/bet-recommendation/{fighter1}/{fighter2}")
def bet_recommendation(
    fighter1: str,
    fighter2: str,
    f1_odds: float,
    f2_odds: float,
    weight_class: str = "Welterweight",
    no_of_rounds: int = 3,
):
    """
    Compute agreement type, Kelly sizing, and GTD parlay signal for a given matchup.
    fighter1 is always treated as the red corner (F1).
    Requires f1_odds and f2_odds as query parameters (American moneyline).
    """
    # ── 1. Look up fighter stats ──────────────────────────────────────────────
    f1_ufc    = get_latest_ufc_stats(fighter1) or {}
    f2_ufc    = get_latest_ufc_stats(fighter2) or {}
    f1_career = get_career_stats(fighter1)
    f2_career = get_career_stats(fighter2)
    f1_extra  = get_fighter_extra_stats(fighter1, weight_class)
    f2_extra  = get_fighter_extra_stats(fighter2, weight_class)
    f1_elo    = get_elo_stats(fighter1)
    f2_elo    = get_elo_stats(fighter2)

    f1_age = float(f1_extra.get('age') or f1_ufc.get('age') or 28)
    f2_age = float(f2_extra.get('age') or f2_ufc.get('age') or 28)

    # total MMA record from fighter_stats_lookup (same as /fighter/ endpoint)
    f1_csv = fighter_stats_lookup.get(fighter1, {})
    f2_csv = fighter_stats_lookup.get(fighter2, {})
    f1_total_wins   = float(f1_csv['wins'])   if f1_csv.get('wins')   is not None else float(f1_ufc.get('wins', 0))
    f1_total_losses = float(f1_csv['losses']) if f1_csv.get('losses') is not None else float(f1_ufc.get('losses', 0))
    f2_total_wins   = float(f2_csv['wins'])   if f2_csv.get('wins')   is not None else float(f2_ufc.get('wins', 0))
    f2_total_losses = float(f2_csv['losses']) if f2_csv.get('losses') is not None else float(f2_ufc.get('losses', 0))

    # ── 2. Build FightInput and run M1 ───────────────────────────────────────
    fight = FightInput(
        F1_wins=float(f1_ufc.get('wins', 0)),
        F1_losses=float(f1_ufc.get('losses', 0)),
        F1_total_wins=f1_total_wins,
        F1_total_losses=f1_total_losses,
        F1_Height_cms=float(f1_ufc.get('Height_cms', 175)),
        F1_Reach_cms=float(f1_extra.get('Reach') or f1_ufc.get('Height_cms', 175)),
        F1_age=f1_age,
        F1_avg_SIG_STR_landed=float(f1_ufc.get('avg_SIG_STR_landed', 0)),
        F1_avg_SIG_STR_pct=float(f1_ufc.get('avg_SIG_STR_pct', 0)),
        F1_avg_TD_landed=float(f1_ufc.get('avg_TD_landed', 0)),
        F1_avg_TD_pct=float(f1_ufc.get('avg_TD_pct', 0)),
        F1_avg_SUB_ATT=float(f1_ufc.get('avg_SUB_ATT', 0)),
        F1_win_by_KO_TKO=float(f1_ufc.get('win_by_KO_TKO', 0)),
        F1_win_by_Submission=float(f1_ufc.get('win_by_Submission', 0)),
        F1_win_by_Decision_Unanimous=float(f1_ufc.get('win_by_Decision_Unanimous', 0)),
        F1_win_by_Decision_Split=float(f1_ufc.get('win_by_Decision_Split', 0)),
        F1_win_by_Decision_Majority=float(f1_ufc.get('win_by_Decision_Majority', 0)),
        F1_current_win_streak=float(f1_ufc.get('current_win_streak', 0)),
        F1_current_lose_streak=float(f1_ufc.get('current_lose_streak', 0)),
        F1_longest_win_streak=float(f1_ufc.get('longest_win_streak', 0)),
        F1_total_title_bouts=float(f1_ufc.get('total_title_bouts', 0)),
        F1_is_southpaw=float(f1_extra.get('is_southpaw', 0)),
        F1_cum_fights=float(f1_career['cum_fights']),
        F1_career_win_rate=float(f1_career['career_win_rate']),
        F1_last5_won=float(f1_career['last5_won']),
        F1_last5_finish_rate=float(f1_career['last5_finish_rate']),
        F1_ko_finish_rate=float(f1_career['ko_finish_rate']),
        F1_sub_finish_rate=float(f1_career['sub_finish_rate']),
        F1_last3_win_rate=float(f1_career['last3_win_rate']),
        F1_last10_win_rate=float(f1_career['last10_win_rate']),
        F1_trend_score=float(f1_career['trend_score']),
        F1_opp_quality=float(f1_career['opp_quality']),
        F1_layoff_days=float(f1_career['layoff_days']),
        F1_SLpM=float(f1_extra.get('SLpM', 0)),
        F1_SApM=float(f1_extra.get('SApM', 0)),
        F1_Str_Acc=float(f1_extra.get('Str_Acc', 0)),
        F1_Str_Def=float(f1_extra.get('Str_Def', 0)),
        F1_TD_Avg=float(f1_extra.get('TD_Avg', 0)),
        F1_TD_Acc=float(f1_extra.get('TD_Acc', 0)),
        F1_TD_Def=float(f1_extra.get('TD_Def', 0)),
        F1_Sub_Avg=float(f1_extra.get('Sub_Avg', 0)),
        F2_wins=float(f2_ufc.get('wins', 0)),
        F2_losses=float(f2_ufc.get('losses', 0)),
        F2_total_wins=f2_total_wins,
        F2_total_losses=f2_total_losses,
        F2_Height_cms=float(f2_ufc.get('Height_cms', 175)),
        F2_Reach_cms=float(f2_extra.get('Reach') or f2_ufc.get('Height_cms', 175)),
        F2_age=f2_age,
        F2_avg_SIG_STR_landed=float(f2_ufc.get('avg_SIG_STR_landed', 0)),
        F2_avg_SIG_STR_pct=float(f2_ufc.get('avg_SIG_STR_pct', 0)),
        F2_avg_TD_landed=float(f2_ufc.get('avg_TD_landed', 0)),
        F2_avg_TD_pct=float(f2_ufc.get('avg_TD_pct', 0)),
        F2_avg_SUB_ATT=float(f2_ufc.get('avg_SUB_ATT', 0)),
        F2_win_by_KO_TKO=float(f2_ufc.get('win_by_KO_TKO', 0)),
        F2_win_by_Submission=float(f2_ufc.get('win_by_Submission', 0)),
        F2_win_by_Decision_Unanimous=float(f2_ufc.get('win_by_Decision_Unanimous', 0)),
        F2_win_by_Decision_Split=float(f2_ufc.get('win_by_Decision_Split', 0)),
        F2_win_by_Decision_Majority=float(f2_ufc.get('win_by_Decision_Majority', 0)),
        F2_current_win_streak=float(f2_ufc.get('current_win_streak', 0)),
        F2_current_lose_streak=float(f2_ufc.get('current_lose_streak', 0)),
        F2_longest_win_streak=float(f2_ufc.get('longest_win_streak', 0)),
        F2_total_title_bouts=float(f2_ufc.get('total_title_bouts', 0)),
        F2_is_southpaw=float(f2_extra.get('is_southpaw', 0)),
        F2_cum_fights=float(f2_career['cum_fights']),
        F2_career_win_rate=float(f2_career['career_win_rate']),
        F2_last5_won=float(f2_career['last5_won']),
        F2_last5_finish_rate=float(f2_career['last5_finish_rate']),
        F2_ko_finish_rate=float(f2_career['ko_finish_rate']),
        F2_sub_finish_rate=float(f2_career['sub_finish_rate']),
        F2_last3_win_rate=float(f2_career['last3_win_rate']),
        F2_last10_win_rate=float(f2_career['last10_win_rate']),
        F2_trend_score=float(f2_career['trend_score']),
        F2_opp_quality=float(f2_career['opp_quality']),
        F2_layoff_days=float(f2_career['layoff_days']),
        F2_SLpM=float(f2_extra.get('SLpM', 0)),
        F2_SApM=float(f2_extra.get('SApM', 0)),
        F2_Str_Acc=float(f2_extra.get('Str_Acc', 0)),
        F2_Str_Def=float(f2_extra.get('Str_Def', 0)),
        F2_TD_Avg=float(f2_extra.get('TD_Avg', 0)),
        F2_TD_Acc=float(f2_extra.get('TD_Acc', 0)),
        F2_TD_Def=float(f2_extra.get('TD_Def', 0)),
        F2_Sub_Avg=float(f2_extra.get('Sub_Avg', 0)),
        F1_elo=float(f1_elo['elo']),
        F2_elo=float(f2_elo['elo']),
        F1_elo_trend=float(f1_elo['elo_trend']),
        F2_elo_trend=float(f2_elo['elo_trend']),
        f1_odds=f1_odds,
        f2_odds=f2_odds,
        weight_class=weight_class,
        no_of_rounds=no_of_rounds,
        f1_name=fighter1,
        f2_name=fighter2,
    )
    m1_result = predict(fight)
    m1_prob   = m1_result['m1_f1_probability'] / 100.0

    # ── 3. Run M2A ────────────────────────────────────────────────────────────
    m2a_input = Model2Input(
        model1_prob=m1_prob,
        f1_odds=f1_odds,
        f2_odds=f2_odds,
        f1_ufc_wins=float(f1_ufc.get('wins', 0)),
        f2_ufc_wins=float(f2_ufc.get('wins', 0)),
        f1_ko_finish_rate=float(f1_career['ko_finish_rate']),
        f2_ko_finish_rate=float(f2_career['ko_finish_rate']),
        f1_sub_finish_rate=float(f1_career['sub_finish_rate']),
        f2_sub_finish_rate=float(f2_career['sub_finish_rate']),
        str_def_dif=float(f1_extra.get('Str_Def', 0)) - float(f2_extra.get('Str_Def', 0)),
        weight_class=weight_class,
        no_of_rounds=no_of_rounds,
    )
    m2a_result = model2a_predict(m2a_input)
    m2a_prob   = m2a_result['m2_prob_f1'] / 100.0

    # ── 4. Agreement type (new four-way classification) ───────────────────────
    f1_raw   = _implied(f1_odds)
    f2_raw   = _implied(f2_odds)
    total    = f1_raw + f2_raw
    f1_novig = f1_raw / total
    f2_novig = f2_raw / total

    m2a_picks_f1 = m2a_prob > 0.5
    m1_picks_f1  = m1_prob  > 0.5
    vegas_fav_f1 = f1_novig > 0.5

    pick_prob_val  = m2a_prob if m2a_picks_f1 else 1.0 - m2a_prob
    pick_novig_val = f1_novig if m2a_picks_f1 else f2_novig
    closing_odds   = f1_odds  if m2a_picks_f1 else f2_odds

    gap      = pick_prob_val - pick_novig_val
    gap_size = abs(gap)
    gap_dir  = 1.0 if gap >= 0 else -1.0

    m1_m2a_agree = int(m1_picks_f1 == m2a_picks_f1)
    vegas_agree  = int(m2a_picks_f1 == vegas_fav_f1)

    if m1_m2a_agree == 0:
        agreement_type = 'SPLIT'
    elif vegas_agree == 0 and gap_dir >= 0:
        agreement_type = 'CONFIRM_DOG'
    elif vegas_agree == 1 and gap_dir >= 0:
        agreement_type = 'CONFIRM_FAV'
    else:
        agreement_type = 'NO_EDGE'

    # ── 5. Quarter Kelly ──────────────────────────────────────────────────────
    if closing_odds > 0:
        dec_odds = closing_odds / 100.0 + 1.0
    else:
        dec_odds = 100.0 / abs(closing_odds) + 1.0
    b            = dec_odds - 1.0
    p            = pick_prob_val
    q            = 1.0 - p
    raw_kelly    = max(0.0, (b * p - q) / b)
    kelly_fraction = round(raw_kelly * 0.25, 4)

    # ── 6. should_bet ─────────────────────────────────────────────────────────
    should_bet = agreement_type in ('CONFIRM_DOG', 'CONFIRM_FAV') and gap_size >= 0.05

    # ── 7. GTD parlay signal (3A) ─────────────────────────────────────────────
    gtd_prob   = None
    gtd_parlay = False
    if MODEL3_AVAILABLE:
        r_cr   = get_career_method_rates(fighter1)
        b_cr   = get_career_method_rates(fighter2)
        wc_ord = WC_ORDER.get(weight_class, 8)
        is_5rnd      = 1 if no_of_rounds >= 5 else 0
        is_womens_flg = 1 if is_womens_fight(weight_class) else 0
        feat3a = {
            'weight_class_ord': wc_ord,
            'is_5rnd':          is_5rnd,
            'is_title':         0,
            'is_womens':        is_womens_flg,
            'R_career_is_finish':        r_cr['career_is_finish'],
            'R_career_is_decision':      r_cr['career_is_decision'],
            'R_career_is_ko':            r_cr['career_is_ko'],
            'R_career_is_sub':           r_cr['career_is_sub'],
            'R_career_finish_delivered': r_cr['career_finish_delivered'],
            'R_career_finish_received':  r_cr['career_finish_received'],
            'R_career_dec_delivered':    r_cr['career_dec_delivered'],
            'R_career_dec_received':     r_cr['career_dec_received'],
            'R_career_n_fights':         r_cr['career_n_fights'],
            'B_career_is_finish':        b_cr['career_is_finish'],
            'B_career_is_decision':      b_cr['career_is_decision'],
            'B_career_is_ko':            b_cr['career_is_ko'],
            'B_career_is_sub':           b_cr['career_is_sub'],
            'B_career_finish_delivered': b_cr['career_finish_delivered'],
            'B_career_finish_received':  b_cr['career_finish_received'],
            'B_career_dec_delivered':    b_cr['career_dec_delivered'],
            'B_career_dec_received':     b_cr['career_dec_received'],
            'B_career_n_fights':         b_cr['career_n_fights'],
            'combined_is_finish':        r_cr['career_is_finish']        + b_cr['career_is_finish'],
            'combined_is_decision':      r_cr['career_is_decision']      + b_cr['career_is_decision'],
            'combined_is_ko':            r_cr['career_is_ko']            + b_cr['career_is_ko'],
            'combined_is_sub':           r_cr['career_is_sub']           + b_cr['career_is_sub'],
            'combined_finish_delivered': r_cr['career_finish_delivered'] + b_cr['career_finish_delivered'],
            'combined_finish_received':  r_cr['career_finish_received']  + b_cr['career_finish_received'],
            'combined_dec_delivered':    r_cr['career_dec_delivered']    + b_cr['career_dec_delivered'],
            'combined_dec_received':     r_cr['career_dec_received']     + b_cr['career_dec_received'],
            'R_SLpM':    float(f1_extra.get('SLpM', 0)),
            'R_SApM':    float(f1_extra.get('SApM', 0)),
            'R_Str_Def': float(f1_extra.get('Str_Def', 0)),
            'R_TD_Avg':  float(f1_extra.get('TD_Avg', 0)),
            'R_TD_Def':  float(f1_extra.get('TD_Def', 0)),
            'R_Sub_Avg': float(f1_extra.get('Sub_Avg', 0)),
            'B_SLpM':    float(f2_extra.get('SLpM', 0)),
            'B_SApM':    float(f2_extra.get('SApM', 0)),
            'B_Str_Def': float(f2_extra.get('Str_Def', 0)),
            'B_TD_Avg':  float(f2_extra.get('TD_Avg', 0)),
            'B_TD_Def':  float(f2_extra.get('TD_Def', 0)),
            'B_Sub_Avg': float(f2_extra.get('Sub_Avg', 0)),
            'combined_SLpM':    float(f1_extra.get('SLpM', 0))    + float(f2_extra.get('SLpM', 0)),
            'combined_SApM':    float(f1_extra.get('SApM', 0))    + float(f2_extra.get('SApM', 0)),
            'combined_Str_Def': float(f1_extra.get('Str_Def', 0)) + float(f2_extra.get('Str_Def', 0)),
            'combined_TD_Avg':  float(f1_extra.get('TD_Avg', 0))  + float(f2_extra.get('TD_Avg', 0)),
            'combined_TD_Def':  float(f1_extra.get('TD_Def', 0))  + float(f2_extra.get('TD_Def', 0)),
            'combined_Sub_Avg': float(f1_extra.get('Sub_Avg', 0)) + float(f2_extra.get('Sub_Avg', 0)),
            'R_avg_SIG_STR_landed':    float(f1_ufc.get('avg_SIG_STR_landed', 0)),
            'B_avg_SIG_STR_landed':    float(f2_ufc.get('avg_SIG_STR_landed', 0)),
            'combined_sig_str_landed': float(f1_ufc.get('avg_SIG_STR_landed', 0)) + float(f2_ufc.get('avg_SIG_STR_landed', 0)),
            'R_avg_TD_landed':         float(f1_ufc.get('avg_TD_landed', 0)),
            'B_avg_TD_landed':         float(f2_ufc.get('avg_TD_landed', 0)),
            'combined_td_landed':      float(f1_ufc.get('avg_TD_landed', 0)) + float(f2_ufc.get('avg_TD_landed', 0)),
            'R_avg_SUB_ATT':           float(f1_ufc.get('avg_SUB_ATT', 0)),
            'B_avg_SUB_ATT':           float(f2_ufc.get('avg_SUB_ATT', 0)),
            'combined_sub_att':        float(f1_ufc.get('avg_SUB_ATT', 0)) + float(f2_ufc.get('avg_SUB_ATT', 0)),
            'reach_dif':       float(f1_extra.get('Reach') or 0) - float(f2_extra.get('Reach') or 0),
            'age_dif':         f1_age - f2_age,
            'sig_str_dif':     float(f1_ufc.get('avg_SIG_STR_landed', 0)) - float(f2_ufc.get('avg_SIG_STR_landed', 0)),
            'avg_sub_att_dif': float(f1_ufc.get('avg_SUB_ATT', 0))        - float(f2_ufc.get('avg_SUB_ATT', 0)),
            'ko_dif':          float(f1_ufc.get('win_by_KO_TKO', 0))      - float(f2_ufc.get('win_by_KO_TKO', 0)),
            'sub_dif':         float(f1_ufc.get('win_by_Submission', 0))  - float(f2_ufc.get('win_by_Submission', 0)),
        }
        df3a = pd.DataFrame([feat3a])[model3a_features]
        p_3a     = 0.30 * model3a_lr.predict_proba(df3a)[0] + 0.70 * model3a_xgb.predict_proba(df3a)[0]
        gtd_prob = float(p_3a[1])
        gtd_parlay = gtd_prob >= 0.50

    # ── 8. Build plain-English bet_notes ─────────────────────────────────────
    pick_name  = fighter1 if m2a_picks_f1 else fighter2
    gap_label  = _ZONE_LABELS[_gap_zone(gap_size)]
    gap_pct    = round(gap_size * 100, 1)

    if agreement_type == 'SPLIT':
        bet_notes = (
            f"M1 and M2A disagree on the winner — no bet. "
            f"M1 favors {'Fighter 1' if m1_picks_f1 else 'Fighter 2'}, "
            f"M2A favors {pick_name}."
        )
    elif agreement_type == 'CONFIRM_DOG':
        base = (
            f"Both models pick {pick_name} (the Vegas underdog) with {gap_pct}pp edge "
            f"over the no-vig line ({gap_label} zone)."
        )
        if not should_bet:
            bet_notes = base + " Gap below 5% threshold — monitor but don't bet."
        elif gtd_parlay and gtd_prob is not None:
            bet_notes = (
                base + f" Strong bet. "
                f"3A predicts {round(gtd_prob*100, 1)}% decision probability — "
                f"consider parlaying with GTD Yes at -110 for added EV."
            )
        else:
            finish_note = f" 3A predicts finish ({round((1-gtd_prob)*100,1)}% finish prob) — skip GTD parlay." if gtd_prob is not None else ""
            bet_notes = base + " Strong bet." + finish_note
    elif agreement_type == 'CONFIRM_FAV':
        base = (
            f"Both models pick {pick_name} (the Vegas favorite) with {gap_pct}pp edge "
            f"over the no-vig line ({gap_label} zone)."
        )
        if not should_bet:
            bet_notes = base + " Gap below 5% threshold — no bet."
        elif gtd_parlay and gtd_prob is not None:
            bet_notes = (
                base + f" Bet. "
                f"3A predicts {round(gtd_prob*100, 1)}% decision probability — "
                f"GTD parlay available but lower EV than underdog plays."
            )
        else:
            finish_note = f" 3A predicts finish — skip GTD parlay." if gtd_prob is not None else ""
            bet_notes = base + " Bet." + finish_note
    else:  # NO_EDGE
        bet_notes = (
            f"Both models pick {pick_name} but M2A is less confident than Vegas "
            f"({gap_pct}pp below no-vig). No edge — do not bet."
        )

    return {
        "fighter1":        fighter1,
        "fighter2":        fighter2,
        "m2a_pick":        pick_name,
        "m1_prob_f1":      round(m1_prob * 100, 1),
        "m2a_prob_pick":   round(pick_prob_val * 100, 1),
        "agreement_type":  agreement_type,
        "should_bet":      should_bet,
        "gap_size_pct":    round(gap_size * 100, 2),
        "gap_zone":        gap_label,
        "kelly_fraction":  kelly_fraction,
        "gtd_parlay":      gtd_parlay,
        "gtd_prob":        round(gtd_prob * 100, 1) if gtd_prob is not None else None,
        "bet_notes":       bet_notes,
    }


@app.get("/odds")
def get_odds():
    # Fetch live DraftKings odds for MMA
    try:
        resp = requests.get(
            'https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/',
            params={
                'apiKey':      _ODDS_API_KEY,
                'regions':     'us',
                'markets':     'h2h',
                'bookmakers':  'draftkings',
            },
            timeout=10,
        )
        resp.raise_for_status()
        api_data = resp.json()
    except Exception as exc:
        return {'error': str(exc), 'fights': []}

    # Match each API event to a card fight by fuzzy name similarity
    timestamp      = datetime.now().isoformat(timespec='seconds')
    snapshot_fights = []

    for event in api_data:
        dk = next((b for b in event.get('bookmakers', []) if b['key'] == 'draftkings'), None)
        if not dk:
            continue
        h2h = next((m for m in dk.get('markets', []) if m['key'] == 'h2h'), None)
        if not h2h or len(h2h.get('outcomes', [])) < 2:
            continue

        outcomes = h2h['outcomes']
        o1_name, o1_price = outcomes[0]['name'], outcomes[0]['price']
        o2_name, o2_price = outcomes[1]['name'], outcomes[1]['price']

        best_score, best_pair, flip = 0.0, None, False
        for card_f1, card_f2 in _CARD_FIGHTS:
            s_normal = _name_sim(o1_name, card_f1) + _name_sim(o2_name, card_f2)
            s_flip   = _name_sim(o1_name, card_f2) + _name_sim(o2_name, card_f1)
            if s_normal > best_score:
                best_score, best_pair, flip = s_normal, (card_f1, card_f2), False
            if s_flip > best_score:
                best_score, best_pair, flip = s_flip, (card_f1, card_f2), True

        if best_score < 0.7 or best_pair is None:
            continue

        card_f1, card_f2 = best_pair
        f1_price = o1_price if not flip else o2_price
        f2_price = o2_price if not flip else o1_price

        snapshot_fights.append({
            'f1': card_f1, 'f2': card_f2,
            'f1_price': f1_price, 'f2_price': f2_price,
        })

    # Load existing snapshots and append new one
    if os.path.exists(_ODDS_SNAPSHOT_FILE):
        with open(_ODDS_SNAPSHOT_FILE) as fp:
            snapshots = json.load(fp)
    else:
        snapshots = []

    snapshots.append({'timestamp': timestamp, 'fights': snapshot_fights})
    with open(_ODDS_SNAPSHOT_FILE, 'w') as fp:
        json.dump(snapshots, fp)

    # Opening snapshot (first ever) for line-movement delta
    opening_map: dict[str, tuple[int, int]] = {}
    for fight in snapshots[0].get('fights', []):
        key = f"{fight['f1']}|{fight['f2']}"
        opening_map[key] = (fight['f1_price'], fight['f2_price'])

    # Build response with movement
    result = []
    for fight in snapshot_fights:
        key     = f"{fight['f1']}|{fight['f2']}"
        opening = opening_map.get(key)
        result.append({
            'f1':            fight['f1'],
            'f2':            fight['f2'],
            'f1_price':      fight['f1_price'],
            'f2_price':      fight['f2_price'],
            'open_f1_price': opening[0] if opening else None,
            'open_f2_price': opening[1] if opening else None,
            'delta_f1':      fight['f1_price'] - opening[0] if opening else None,
            'delta_f2':      fight['f2_price'] - opening[1] if opening else None,
        })

    return {'timestamp': timestamp, 'fights': result}
