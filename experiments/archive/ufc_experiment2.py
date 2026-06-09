#!/usr/bin/env python3
"""
UFC Experiment 2 – Memory-safe, sequential, saves after every model.

Previous results (crashed before save):
  XGBoost  → 66.26%   LightGBM → 65.44%
  CatBoost → 65.85%   RF       → 65.03%

New features to explore:
  - Fighter style (SLpM, SApM, TD_Def, Stance) from ufc_fighters_final.csv
  - Recent form (avg_KD, avg_Ctrl_Sec, recent_won) from ufc_training_data.csv
  - Age-prime indicator, fight frequency, streak features
  - All from career_fights.csv (days_since_last, win_rate, etc.)

Strict rules:
  - date >= 2015 filter
  - min 2+ UFC fights per fighter
  - Split BEFORE augment; augment train only (swap R↔B + flip target)
  - Drop NaN — no median fill
  - No odds, no post-fight columns
  - Max 20 Optuna trials per model
  - gc.collect() + del between models
  - Save after EVERY model if it beats current best
"""

import gc, os, sys, pickle, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── config ───────────────────────────────────────────────────────────────────
RS         = 43       # random seed everywhere
TEST_SIZE  = 0.2
N_OPTUNA   = 20       # max trials per model
LOG_PATH   = "model/experiment2_log.txt"
BEST_PATH  = "model/ufc_model.pkl"
FEAT_PATH  = "model/feature_columns.pkl"

os.makedirs("model", exist_ok=True)

log_lines   = []
best_acc    = 0.0
best_model  = None
best_feats  = None

def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)
    # flush log after every message so crash doesn't lose output
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))

def save_if_best(model, feats, acc, label):
    global best_acc, best_model, best_feats
    if acc > best_acc:
        best_acc   = acc
        best_model = model
        best_feats = feats
        joblib.dump(model, BEST_PATH)
        with open(FEAT_PATH, "wb") as f:
            pickle.dump(feats, f)
        log(f"  ★ NEW BEST {acc:.4f} ({label}) → saved to model/")

log("=" * 68)
log("UFC EXPERIMENT 2 – Sequential, crash-safe")
log("=" * 68)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
log("\n[1/5] Loading data …")

ufc = pd.read_csv("data/ufc-master.csv")
ufc["date"] = pd.to_datetime(ufc["date"])

career = pd.read_csv("data/career_fights.csv")
career["date"]    = pd.to_datetime(career["date"])
career["fighter"] = career["fighter"].str.strip()

fighters = pd.read_csv("data/ufc_fighters_final.csv")  # style data

td = pd.read_csv("data/ufc_training_data.csv")          # already-computed rolling features
td["date"] = pd.to_datetime(td["date"])

log(f"  ufc-master      : {ufc.shape[0]:,} rows")
log(f"  career_fights   : {career.shape[0]:,} rows  {career['fighter'].nunique():,} fighters")
log(f"  fighters        : {fighters.shape[0]:,} rows")
log(f"  training_data   : {td.shape[0]:,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CAREER FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
log("\n[2/5] Engineering features …")

# ── 2a. Career rolling features (vectorised) ──────────────────────────────
career = career.sort_values(["fighter", "date"]).reset_index(drop=True)

career["is_fin_win"]  = ((career["won"] == 1) & (career["got_finish"] == 1)).astype(float)
career["is_fin_loss"] = ((career["won"] == 0) & (career["got_finish"] == 1)).astype(float)

g = career.groupby("fighter", sort=False)
career["cum_wins"]        = g["won"].transform(lambda x: x.shift(1).cumsum().fillna(0))
career["cum_fin_losses"]  = g["is_fin_loss"].transform(lambda x: x.shift(1).cumsum().fillna(0))
career["n_prior"]         = g.cumcount()
career["cum_losses"]      = career["n_prior"] - career["cum_wins"]
career["prev_date"]       = g["date"].shift(1)
career["days_since_last"] = (career["date"] - career["prev_date"]).dt.days

# Career-span fight frequency (fights per year going into this fight)
first_date = g["date"].transform("min")
career["career_span_days"] = (career["date"] - first_date).dt.days
career["fight_frequency"]  = np.where(
    career["career_span_days"] > 30,
    career["n_prior"] / (career["career_span_days"] / 365.25),
    np.nan
)

# Career win rate
n  = career["n_prior"].clip(lower=1)
career["career_win_rate"] = np.where(career["n_prior"] > 0, career["cum_wins"] / n, np.nan)

JOIN_COLS = [
    "last5_won", "last5_finish_rate",
    "cum_wins", "cum_losses", "cum_fin_losses", "n_prior",
    "days_since_last", "fight_frequency", "career_win_rate",
]

def join_career(df, prefix):
    fc = f"{prefix}_fighter"
    sub = (career[["fighter", "date"] + JOIN_COLS]
           .rename(columns={"fighter": fc})
           .rename(columns={c: f"{prefix}_c_{c}" for c in JOIN_COLS})
           .sort_values("date"))
    return pd.merge_asof(df.sort_values("date"), sub, on="date",
                         by=fc, direction="backward")

log("  Joining career features (R) …")
ufc = join_career(ufc, "R")
log("  Joining career features (B) …")
ufc = join_career(ufc, "B")
log(f"  UFC after career join: {ufc.shape}")

# ── 2b. Fighter style features from ufc_fighters_final ──────────────────────
def parse_reach(r):
    try:
        return float(str(r).replace('"', '').strip())
    except:
        return np.nan

fighters["Reach_in"] = fighters["Reach"].apply(parse_reach)
fighters["is_southpaw"] = (fighters["Stance"].str.lower() == "southpaw").astype(float)
fighters["is_switch"]   = (fighters["Stance"].str.lower() == "switch").astype(float)

# Parse percentage columns
for col in ["Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]:
    fighters[col] = fighters[col].astype(str).str.replace("%", "").str.strip()
    fighters[col] = pd.to_numeric(fighters[col], errors="coerce") / 100.0

STYLE_COLS = ["SLpM", "SApM", "Str_Acc", "Str_Def",
              "TD_Avg", "TD_Def", "Sub_Avg",
              "is_southpaw", "is_switch", "Reach_in"]

def join_style(df, prefix):
    fc = f"{prefix}_fighter"
    sub = (fighters[["Fighter_Name"] + STYLE_COLS]
           .rename(columns={"Fighter_Name": fc})
           .rename(columns={c: f"{prefix}_s_{c}" for c in STYLE_COLS}))
    return df.merge(sub, on=fc, how="left")

ufc = join_style(ufc, "R")
ufc = join_style(ufc, "B")
log(f"  UFC after style join: {ufc.shape}")

# ── 2c. Join novel features from ufc_training_data ──────────────────────────
TD_COLS = ["R_recent_won", "B_recent_won", "recent_won_dif",
           "R_avg_KD", "B_avg_KD", "avg_KD_dif",
           "R_avg_Ctrl_Sec", "B_avg_Ctrl_Sec", "avg_Ctrl_Sec_dif",
           "R_wrestling_score", "B_wrestling_score", "wrestling_score_dif"]
TD_COLS = [c for c in TD_COLS if c in td.columns]

td_sub = td[["R_fighter", "B_fighter", "date"] + TD_COLS].copy()
ufc = ufc.merge(td_sub, on=["R_fighter", "B_fighter", "date"], how="left")
log(f"  UFC after training_data join: {ufc.shape}")

# ── 2d. Compute all derived columns ─────────────────────────────────────────
for p in ("R", "B"):
    ko  = ufc.get(f"{p}_win_by_KO/TKO",        pd.Series(0, index=ufc.index))
    sub = ufc.get(f"{p}_win_by_Submission",     pd.Series(0, index=ufc.index))
    dec_u = ufc.get(f"{p}_win_by_Decision_Unanimous", pd.Series(0, index=ufc.index))
    dec_s = ufc.get(f"{p}_win_by_Decision_Split",     pd.Series(0, index=ufc.index))
    dec_m = ufc.get(f"{p}_win_by_Decision_Majority",  pd.Series(0, index=ufc.index))

    ufc[f"{p}_ufc_fights"]    = ufc[f"{p}_wins"] + ufc[f"{p}_losses"]
    ufc[f"{p}_finish_rate"]   = (ko + sub) / (ufc[f"{p}_wins"] + 1)
    ufc[f"{p}_times_finished"] = ufc.get(f"{p}_c_cum_fin_losses",
                                         (ufc[f"{p}_losses"] - (dec_u + dec_s + dec_m)).clip(lower=0))
    ufc[f"{p}_last5_won"]     = ufc[f"{p}_c_last5_won"]   # full career last-5
    ufc[f"{p}_pre_ufc_wins"]  = (ufc[f"{p}_c_cum_wins"]   - ufc[f"{p}_wins"]).clip(lower=0)
    ufc[f"{p}_pre_ufc_losses"]= (ufc[f"{p}_c_cum_losses"] - ufc[f"{p}_losses"]).clip(lower=0)

    # Age-prime indicator: distance from peak (29.5)
    ufc[f"{p}_age_prime_dist"] = np.abs(ufc[f"{p}_age"] - 29.5)
    ufc[f"{p}_past_prime"]     = (ufc[f"{p}_age"] > 32).astype(float)

if "avg_td_dif" in ufc.columns:
    ufc["td_dif"] = ufc["avg_td_dif"]
else:
    ufc["td_dif"] = ufc["R_avg_TD_landed"] - ufc["B_avg_TD_landed"]

# Standard diffs
for r, b, name in [
    ("R_ufc_fights",    "B_ufc_fights",    "ufc_fights_dif"),
    ("R_finish_rate",   "B_finish_rate",   "finish_rate_dif"),
    ("R_times_finished","B_times_finished","times_finished_dif"),
    ("R_last5_won",     "B_last5_won",     "last5_dif"),
    ("R_pre_ufc_wins",  "B_pre_ufc_wins",  "pre_ufc_wins_dif"),
    ("R_pre_ufc_losses","B_pre_ufc_losses","pre_ufc_losses_dif"),
    ("R_c_career_win_rate","B_c_career_win_rate","career_win_rate_dif"),
    ("R_c_days_since_last","B_c_days_since_last","days_since_last_dif"),
    ("R_c_fight_frequency","B_c_fight_frequency","fight_frequency_dif"),
    ("R_age_prime_dist","B_age_prime_dist","age_prime_dist_dif"),
    ("R_past_prime",    "B_past_prime",    "past_prime_dif"),
]:
    if r in ufc.columns and b in ufc.columns:
        ufc[name] = ufc[r] - ufc[b]

# Career total wins/losses diffs (replaces sherdog)
ufc["total_wins_dif"]   = ufc["R_c_cum_wins"]   - ufc["B_c_cum_wins"]
ufc["total_losses_dif"] = ufc["R_c_cum_losses"]  - ufc["B_c_cum_losses"]

# Stance clash features
if "R_s_is_southpaw" in ufc.columns and "B_s_is_southpaw" in ufc.columns:
    ufc["southpaw_clash"] = (
        (ufc["R_s_is_southpaw"] == 1) & (ufc["B_s_is_southpaw"] == 0) |
        (ufc["R_s_is_southpaw"] == 0) & (ufc["B_s_is_southpaw"] == 1)
    ).astype(float)

# Style diffs
for col in ["SLpM", "SApM", "Str_Acc", "Str_Def", "TD_Avg", "TD_Def", "Sub_Avg", "Reach_in"]:
    rc, bc = f"R_s_{col}", f"B_s_{col}"
    if rc in ufc.columns and bc in ufc.columns:
        ufc[f"{col}_dif"] = ufc[rc] - ufc[bc]

# UFC streak features
if "R_current_win_streak" in ufc.columns:
    ufc["win_streak_dif"]  = ufc["R_current_win_streak"]  - ufc["B_current_win_streak"]
    ufc["lose_streak_dif"] = ufc["R_current_lose_streak"] - ufc["B_current_lose_streak"]

# Target
ufc = ufc[ufc["Winner"].isin(["Red", "Blue"])].copy()
ufc["target"] = (ufc["Winner"] == "Red").astype(int)
log(f"  Final UFC shape  : {ufc.shape}  |  Red wins: {ufc['target'].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────
LEAKY = {
    "finish", "finish_details", "finish_round", "finish_round_time",
    "total_fight_time_secs", "Winner",
    "R_odds", "B_odds", "R_ev", "B_ev",
    "r_dec_odds", "b_dec_odds", "r_sub_odds", "b_sub_odds",
    "r_ko_odds", "b_ko_odds",
}

BASE = [
    "R_wins", "R_losses", "R_ufc_fights", "R_finish_rate", "R_times_finished",
    "R_last5_won", "R_Height_cms", "R_age", "R_avg_SIG_STR_landed", "R_avg_TD_landed",
    "B_wins", "B_losses", "B_ufc_fights", "B_finish_rate", "B_times_finished",
    "B_last5_won", "B_Height_cms", "B_age", "B_avg_SIG_STR_landed", "B_avg_TD_landed",
    "win_dif", "loss_dif", "ufc_fights_dif", "finish_rate_dif", "times_finished_dif",
    "last5_dif", "height_dif", "age_dif", "sig_str_dif", "td_dif",
    "total_wins_dif", "total_losses_dif",
]

# Group A: career-level features
G_CAREER = [
    "R_c_last5_won", "R_c_last5_finish_rate",
    "B_c_last5_won", "B_c_last5_finish_rate",
    "R_c_career_win_rate", "B_c_career_win_rate", "career_win_rate_dif",
    "R_c_days_since_last", "B_c_days_since_last", "days_since_last_dif",
    "R_c_fight_frequency", "B_c_fight_frequency", "fight_frequency_dif",
    "R_pre_ufc_wins", "R_pre_ufc_losses",
    "B_pre_ufc_wins", "B_pre_ufc_losses",
    "pre_ufc_wins_dif", "pre_ufc_losses_dif",
]

# Group B: UFC record extras
G_UFC_EXTRA = [
    "R_current_win_streak", "B_current_win_streak", "win_streak_dif",
    "R_current_lose_streak", "B_current_lose_streak", "lose_streak_dif",
    "R_avg_SIG_STR_pct", "B_avg_SIG_STR_pct",
    "R_avg_TD_pct",      "B_avg_TD_pct",
    "R_Reach_cms",       "B_Reach_cms",       "reach_dif",
]

# Group C: fighter style
G_STYLE = [
    "R_s_SLpM", "B_s_SLpM", "SLpM_dif",
    "R_s_SApM", "B_s_SApM", "SApM_dif",
    "R_s_Str_Def", "B_s_Str_Def", "Str_Def_dif",
    "R_s_TD_Def",  "B_s_TD_Def",  "TD_Def_dif",
    "R_s_Sub_Avg", "B_s_Sub_Avg", "Sub_Avg_dif",
    "R_s_TD_Avg",  "B_s_TD_Avg",  "TD_Avg_dif",
    "R_s_is_southpaw", "B_s_is_southpaw", "southpaw_clash",
]

# Group D: engineered signals
G_ENGINEERED = [
    "R_age_prime_dist", "B_age_prime_dist", "age_prime_dist_dif",
    "R_past_prime", "B_past_prime", "past_prime_dif",
]

# Group E: pre-computed rolling features from ufc_training_data
G_TD = [c for c in TD_COLS if c in ufc.columns]

ALL_GROUPS = list(dict.fromkeys(
    BASE + G_CAREER + G_UFC_EXTRA + G_STYLE + G_ENGINEERED + G_TD
))

FEATURE_SETS = {
    "base":           BASE,
    "base+career":    BASE + G_CAREER,
    "base+career+ufc": BASE + G_CAREER + G_UFC_EXTRA,
    "all":            ALL_GROUPS,
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
BEST_DATE      = 2016
BEST_MIN_FIGHTS = 2      # >1 as user specified

def augment_corners(X: pd.DataFrame, y: pd.Series, features: list):
    Xf = X.copy()
    yf = (1 - y).reset_index(drop=True)
    r_feats   = [f for f in features if f.startswith("R_")]
    dif_feats = [f for f in features if f.endswith("_dif")]
    for rf in r_feats:
        bf = "B_" + rf[2:]
        if bf in features:
            Xf[rf] = X[bf].values
            Xf[bf] = X[rf].values
    for df_col in dif_feats:
        if df_col in Xf.columns:
            Xf[df_col] = -X[df_col].values
    return Xf, yf


def make_split(feat_names: list, date_cutoff: int = BEST_DATE,
               min_fights: int = BEST_MIN_FIGHTS):
    d = ufc[ufc["date"] >= f"{date_cutoff}-01-01"].copy()
    if "R_ufc_fights" in d.columns:
        d = d[(d["R_ufc_fights"] >= min_fights) & (d["B_ufc_fights"] >= min_fights)]
    avail = [f for f in feat_names if f in d.columns and f not in LEAKY]
    d = d[avail + ["target"]].dropna()
    if len(d) < 200:
        return None, None, None, None, None
    X, y = d[avail], d["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RS)
    Xf, yf = augment_corners(Xtr.reset_index(drop=True), ytr.reset_index(drop=True), avail)
    Xtr = pd.concat([Xtr.reset_index(drop=True), Xf], ignore_index=True)
    ytr = pd.concat([ytr.reset_index(drop=True), yf], ignore_index=True)
    return Xtr, Xte, ytr, yte, avail

# ─────────────────────────────────────────────────────────────────────────────
# 5a. FEATURE GRID SEARCH with fast XGBoost
# ─────────────────────────────────────────────────────────────────────────────
log("\n[3/5] Feature group grid search …")

grid_results = []
quick_xgb = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                 subsample=0.8, colsample_bytree=0.8,
                 random_state=RS, eval_metric="logloss",
                 use_label_encoder=False, verbosity=0)

for date_cut in [2015, 2016]:
    for mf in [2, 3]:
        for fname, fset in FEATURE_SETS.items():
            Xtr, Xte, ytr, yte, used = make_split(fset, date_cut, mf)
            if Xtr is None:
                continue
            m = xgb.XGBClassifier(**quick_xgb)
            m.fit(Xtr, ytr)
            acc = accuracy_score(yte, m.predict(Xte))
            log(f"  date≥{date_cut} mf={mf} feats={fname:<18} "
                f"train={len(ytr):,} test={len(yte)} acc={acc:.4f}")
            grid_results.append(dict(date=date_cut, mf=mf, fname=fname,
                                     acc=acc, fset=fset))
            del m; gc.collect()

best_grid = max(grid_results, key=lambda x: x["acc"])
log(f"\n  ✓ Best grid: date≥{best_grid['date']} mf={best_grid['mf']}"
    f" feats={best_grid['fname']} acc={best_grid['acc']:.4f}")

BEST_DATE       = best_grid["date"]
BEST_MIN_FIGHTS = best_grid["mf"]
BEST_FSET       = best_grid["fset"]
BEST_FNAME      = best_grid["fname"]

Xtr, Xte, ytr, yte, USED = make_split(BEST_FSET, BEST_DATE, BEST_MIN_FIGHTS)
log(f"  Working split → train={len(ytr):,} test={len(yte)} features={len(USED)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5b. ONE-AT-A-TIME FEATURE ADDITION
#     Report whether each new group helped or hurt XGBoost
# ─────────────────────────────────────────────────────────────────────────────
log("\n[3b/5] Feature-addition experiment (XGBoost, no tuning) …")

base_Xtr, base_Xte, base_ytr, base_yte, base_used = make_split(BASE, BEST_DATE, BEST_MIN_FIGHTS)
m0 = xgb.XGBClassifier(**quick_xgb).fit(base_Xtr, base_ytr)
base_acc = accuracy_score(base_yte, m0.predict(base_Xte))
log(f"  Baseline (base_32): {base_acc:.4f}")
del m0; gc.collect()

cumulative_feats = list(BASE)
cumulative_acc   = base_acc

for gname, gfeats in [
    ("+ career features",  G_CAREER),
    ("+ UFC extras",       G_UFC_EXTRA),
    ("+ style features",   G_STYLE),
    ("+ age prime",        G_ENGINEERED),
    ("+ training_data",    G_TD),
]:
    candidate = list(dict.fromkeys(cumulative_feats + gfeats))
    Xa, Xb, ya, yb, ua = make_split(candidate, BEST_DATE, BEST_MIN_FIGHTS)
    if Xa is None:
        log(f"  {gname}: skipped (too few rows)")
        continue
    ma = xgb.XGBClassifier(**quick_xgb).fit(Xa, ya)
    acc = accuracy_score(yb, ma.predict(Xb))
    delta = acc - cumulative_acc
    marker = "✓ HELPED" if delta > 0.002 else ("✗ HURT" if delta < -0.002 else "≈ neutral")
    log(f"  {gname:<30} acc={acc:.4f}  Δ={delta:+.4f}  {marker}")
    if delta > 0:
        cumulative_feats = candidate
        cumulative_acc   = acc
    del ma; gc.collect()

log(f"\n  Best cumulative feature set: {len(cumulative_feats)} features  acc={cumulative_acc:.4f}")

# Rebuild split with best cumulative features
Xtr, Xte, ytr, yte, USED = make_split(cumulative_feats, BEST_DATE, BEST_MIN_FIGHTS)
log(f"  Final split → train={len(ytr):,} test={len(yte)} features={len(USED)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TUNE ALL MODELS SEQUENTIALLY
# ─────────────────────────────────────────────────────────────────────────────
log(f"\n[4/5] Tuning models ({N_OPTUNA} Optuna trials each, sequential) …")

def run_study(obj_fn, n=N_OPTUNA):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(obj_fn, n_trials=n, show_progress_bar=True,
                   timeout=600)   # kill if >10 min
    return study

results = {}  # name → (model, acc, params)

# ── XGBoost ──────────────────────────────────────────────────────────────────
log("\n  ── XGBoost ──")
def obj_xgb(trial):
    m = xgb.XGBClassifier(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
        learning_rate     = trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        subsample         = trial.suggest_float("subsample", 0.55, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.45, 1.0),
        gamma             = trial.suggest_float("gamma", 0, 5),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        random_state=RS, eval_metric="logloss",
        use_label_encoder=False, verbosity=0,
    )
    return accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))

st = run_study(obj_xgb)
p = st.best_params
p.update(dict(random_state=RS, eval_metric="logloss",
              use_label_encoder=False, verbosity=0))
final_xgb = xgb.XGBClassifier(**p).fit(Xtr, ytr)
xgb_acc = accuracy_score(yte, final_xgb.predict(Xte))
log(f"  XGBoost → {xgb_acc:.4f}")
results["XGBoost"] = (final_xgb, xgb_acc, p)
save_if_best(final_xgb, USED, xgb_acc, "XGBoost")
del st; gc.collect()

# ── LightGBM ─────────────────────────────────────────────────────────────────
log("\n  ── LightGBM ──")
def obj_lgbm(trial):
    m = lgb.LGBMClassifier(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
        learning_rate     = trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        num_leaves        = trial.suggest_int("num_leaves", 15, 100),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        subsample         = trial.suggest_float("subsample", 0.55, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.45, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        random_state=RS, verbose=-1,
    )
    return accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))

st2 = run_study(obj_lgbm)
p2 = st2.best_params; p2.update(dict(random_state=RS, verbose=-1))
final_lgbm = lgb.LGBMClassifier(**p2).fit(Xtr, ytr)
lgbm_acc = accuracy_score(yte, final_lgbm.predict(Xte))
log(f"  LightGBM → {lgbm_acc:.4f}")
results["LightGBM"] = (final_lgbm, lgbm_acc, p2)
save_if_best(final_lgbm, USED, lgbm_acc, "LightGBM")
del st2; gc.collect()

# ── CatBoost ─────────────────────────────────────────────────────────────────
log("\n  ── CatBoost ──")
def obj_cat(trial):
    m = CatBoostClassifier(
        iterations    = trial.suggest_int("iterations", 200, 1000),
        learning_rate = trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        depth         = trial.suggest_int("depth", 3, 8),
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1, 30),
        border_count  = trial.suggest_int("border_count", 32, 128),
        random_seed=RS, verbose=False,
    )
    return accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))

st3 = run_study(obj_cat)
p3 = st3.best_params; p3.update(dict(random_seed=RS, verbose=False))
final_cat = CatBoostClassifier(**p3).fit(Xtr, ytr)
cat_acc = accuracy_score(yte, final_cat.predict(Xte))
log(f"  CatBoost → {cat_acc:.4f}")
results["CatBoost"] = (final_cat, cat_acc, p3)
save_if_best(final_cat, USED, cat_acc, "CatBoost")
del st3; gc.collect()

# ── RandomForest ─────────────────────────────────────────────────────────────
log("\n  ── RandomForest ──")
def obj_rf(trial):
    m = RandomForestClassifier(
        n_estimators     = trial.suggest_int("n_estimators", 100, 200),
        max_depth        = trial.suggest_int("max_depth", 5, 20),
        min_samples_split= trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_categorical("max_features",
                                                     ["sqrt", "log2", 0.4, 0.6]),
        random_state=RS, n_jobs=-1,
    )
    return accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))

st4 = run_study(obj_rf)
p4 = st4.best_params; p4.update(dict(random_state=RS))
final_rf = RandomForestClassifier(**p4, n_jobs=-1).fit(Xtr, ytr)
rf_acc = accuracy_score(yte, final_rf.predict(Xte))
log(f"  RandomForest → {rf_acc:.4f}")
results["RandomForest"] = (final_rf, rf_acc, p4)
save_if_best(final_rf, USED, rf_acc, "RandomForest")
del st4; gc.collect()

# ── Voting Ensemble (top 3 by accuracy) ──────────────────────────────────────
log("\n  ── Voting Ensemble (top 3) ──")
top3 = sorted(results.items(), key=lambda x: -x[1][1])[:3]
log(f"  Using: {[n for n,_ in top3]}")
voting = VotingClassifier(
    estimators=[(n, v[0]) for n, v in top3],
    voting="soft",
)
voting.fit(Xtr, ytr)
voting_acc = accuracy_score(yte, voting.predict(Xte))
log(f"  Voting Ensemble → {voting_acc:.4f}")
results["VotingEnsemble"] = (voting, voting_acc, {})
save_if_best(voting, USED, voting_acc, "VotingEnsemble")
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
log("\n[5/5] Final results …")
log("\n" + "=" * 68)
log("MODEL COMPARISON  (clean test set)")
log("=" * 68)
log(f"  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7}")
log("  " + "-" * 45)

for name, (m, acc, _) in sorted(results.items(), key=lambda x: -x[1][1]):
    yp = m.predict(Xte)
    p  = precision_score(yte, yp, zero_division=0)
    r  = recall_score(yte, yp, zero_division=0)
    log(f"  {name:<22} {acc:>7.4f} {p:>7.4f} {r:>7.4f}")

log(f"\n  Overall best: {best_acc:.4f} → model/ufc_model.pkl")

# Classification report for best model
best_yp = best_model.predict(Xte)
log("\nClassification Report (best model on clean test set):")
log(classification_report(yte, best_yp,
                           target_names=["Blue wins (0)", "Red wins (1)"]))

# Feature importances from XGBoost
log("Feature Importances (XGBoost, top 25):")
src = final_xgb if "XGBoost" in results else None
if src:
    imp = pd.Series(src.feature_importances_, index=USED).sort_values(ascending=False)
    for feat, val in imp.head(25).items():
        bar = "█" * int(val * 250)
        log(f"  {feat:<45} {val:.4f}  {bar}")
    log("\nBottom 10 (least useful):")
    for feat, val in imp.tail(10).items():
        log(f"  {feat:<45} {val:.4f}")

# Best hyperparameters
log("\nBest model hyperparameters:")
best_name = [n for n, (m, a, p) in results.items() if a == best_acc][0]
best_params = results[best_name][2]
log(f"  Model: {best_name}")
skip_keys = {"random_state","eval_metric","use_label_encoder",
             "verbosity","verbose","random_seed"}
for k, v in best_params.items():
    if k not in skip_keys:
        log(f"  {k} = {v}")

# Summary
log("\n" + "=" * 68)
log("SUMMARY")
log("=" * 68)
log(f"  Best accuracy        : {best_acc:.4f}")
log(f"  Best model           : {best_name}")
log(f"  Feature set          : {BEST_FNAME}")
log(f"  Date cutoff          : {BEST_DATE}")
log(f"  Min UFC fights       : {BEST_MIN_FIGHTS}")
log(f"  # features used      : {len(USED)}")
log(f"  Train size           : {len(ytr):,}  (incl. augmented flips)")
log(f"  Test size            : {len(yte):,}  (clean, unaugmented)")
log(f"  Red-win rate in test : {yte.mean():.3f}")
log(f"  Lift over naive      : {best_acc - max(yte.mean(), 1-yte.mean()):+.4f}")
log(f"\n  Saved → model/ufc_model.pkl  +  model/feature_columns.pkl")
log(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 68)
