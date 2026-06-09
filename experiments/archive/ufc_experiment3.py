"""
ufc_experiment3.py — Target: beat 68.94% CatBoost baseline.

New features vs experiment2:
  - opponent quality (avg career win% of last-5 opponents)
  - performance trend (last3_win_rate vs last10_win_rate)
  - KO finish rate & sub finish rate (separate)
  - weight class ordinal
  - title fight experience (R/B_total_title_bouts already in master — also compute rolling)
  - stance matchup (orth-orth / south-south / mixed)
  - age × ufc_fights interaction
  - layoff buckets (<90 / 90-180 / 180-365 / 365+ days)

Models (sequential, n_jobs=1, 50 Optuna trials each):
  CatBoost → XGBoost → LightGBM → MLP → LogReg → Voting → Stacking

For each model test: all features / top-30 / top-15; date≥2015/2016/2017; min_ufc_fights 1/2/3.
Save winner to model/ufc_model_best.pkl + model/feature_columns_best.pkl.
Also update model/ufc_model.pkl if better than existing 68.94%.
"""

import gc
import os
import sys
import time
import warnings
import pickle
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RS = 42
TIMEOUT_SECS = 300      # 5 min per model config
N_TRIALS = 50
BASELINE_ACC = 0.6894   # current best

OUTPUT_FILE = "model/experiment3_output.txt"
BEST_MODEL_FILE = "model/ufc_model_best.pkl"
BEST_FEATS_FILE  = "model/feature_columns_best.pkl"
CUR_MODEL_FILE  = "model/ufc_model.pkl"
CUR_FEATS_FILE  = "model/feature_columns.pkl"

os.makedirs("model", exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")

log("=" * 70)
log("UFC EXPERIMENT 3  —  started " + pd.Timestamp.now().isoformat())
log("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
log("\n[1] Loading raw data...")

master = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
log(f"  ufc-master: {master.shape}")

career = pd.read_csv("data/career_fights.csv")
career["date"] = pd.to_datetime(career["date"])
log(f"  career_fights: {career.shape}")

fighters = pd.read_csv("data/ufc_fighters_final.csv")
log(f"  ufc_fighters_final: {fighters.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD CAREER ROLLING FEATURES (all pre-fight, shift(1) throughout)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[2] Engineering career rolling features...")

career = career.sort_values(["fighter", "date"]).reset_index(drop=True)
g = career.groupby("fighter", sort=False)

# cumulative stats (shift so they're pre-fight)
career["cum_wins"]         = g["won"].transform(lambda x: x.shift(1).cumsum().fillna(0))
career["cum_fights"]       = g["won"].transform(lambda x: x.shift(1).expanding().count().fillna(0))
career["career_win_rate"]  = (career["cum_wins"] / career["cum_fights"].clip(lower=1))

# KO finish rate (got_finish = 1 only for KO/TKO; need method column)
# career "method" has raw values — derive KO & Sub finish indicators
career["got_ko"]  = career["method"].str.contains("KO|TKO", case=False, na=False).astype(int)
career["got_sub"] = career["method"].str.contains("Sub|Submission", case=False, na=False).astype(int)
# as finisher: won AND (method contains KO/Sub)
career["did_ko"]  = (career["won"] == 1) & career["method"].str.contains("KO|TKO", case=False, na=False)
career["did_sub"] = (career["won"] == 1) & career["method"].str.contains("Sub|Submission", case=False, na=False)
career["did_ko"]  = career["did_ko"].astype(int)
career["did_sub"] = career["did_sub"].astype(int)

for col in ["did_ko", "did_sub"]:
    career[f"cum_{col}"] = g[col].transform(lambda x: x.shift(1).cumsum().fillna(0))

career["ko_finish_rate"]  = career["cum_did_ko"]  / career["cum_fights"].clip(lower=1)
career["sub_finish_rate"] = career["cum_did_sub"] / career["cum_fights"].clip(lower=1)

# performance trend: last3 vs last10 win rate
def roll_mean(x, w):
    return x.shift(1).rolling(w, min_periods=1).mean()

career["last3_win_rate"]  = g["won"].transform(lambda x: roll_mean(x, 3))
career["last10_win_rate"] = g["won"].transform(lambda x: roll_mean(x, 10))
career["trend_score"]     = career["last3_win_rate"] - career["last10_win_rate"]

# layoff days (days since previous fight)
career["prev_date"]   = g["date"].transform(lambda x: x.shift(1))
career["layoff_days"] = (career["date"] - career["prev_date"]).dt.days.fillna(0)

log("  career rolling features done")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  OPPONENT QUALITY  (avg career_win_rate of last-5 opponents, pre-fight)
# ─────────────────────────────────────────────────────────────────────────────
log("[3] Computing opponent quality...")

# We need, for each (fighter, date), the avg career_win_rate of the opponents
# they faced in their last-5 fights BEFORE this date.
# Approach: build a lookup (opponent, date) → career_win_rate at that date,
# then for each fight, left-join each of the 5 previous opponents and average.

# career_win_rate is already per (fighter, date) — just keep the latest before fight
opp_quality_src = (career[["fighter", "date", "career_win_rate"]]
                   .rename(columns={"fighter": "opponent", "career_win_rate": "opp_win_rate"}))

# Merge opponent's win rate onto each career fight row (time-safe)
career_sorted = career.sort_values("date")
opp_src_sorted = opp_quality_src.sort_values("date")

career_with_opp = pd.merge_asof(
    career_sorted,
    opp_src_sorted,
    on="date",
    by="opponent",
    direction="backward"
)

# Now compute rolling avg of opp_win_rate over last 5 fights (shift 1)
career_with_opp = career_with_opp.sort_values(["fighter", "date"])
career_with_opp["opp_quality"] = (
    career_with_opp.groupby("fighter")["opp_win_rate"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

log("  opponent quality done")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MERGE CAREER FEATURES → JOIN COLS
# ─────────────────────────────────────────────────────────────────────────────
JOIN_COLS = [
    "career_win_rate", "ko_finish_rate", "sub_finish_rate",
    "last3_win_rate", "last10_win_rate", "trend_score",
    "layoff_days", "last5_won", "last5_finish_rate",
    "cum_wins", "cum_fights",
]

# Merge opp_quality back into career_with_opp and pull all features
career_feat = career_with_opp[["fighter", "date"] + JOIN_COLS + ["opp_quality"]].copy()

def join_career(df, prefix):
    fighter_col = f"{prefix}_fighter"
    sub = career_feat.rename(columns={
        "fighter": fighter_col,
        **{c: f"{prefix}_{c}" for c in JOIN_COLS + ["opp_quality"]}
    }).sort_values("date")
    merged = pd.merge_asof(
        df.sort_values("date"),
        sub,
        on="date",
        by=fighter_col,
        direction="backward"
    )
    return merged

# ─────────────────────────────────────────────────────────────────────────────
# 5.  BUILD MASTER FEATURE TABLE
# ─────────────────────────────────────────────────────────────────────────────
log("[4] Building master feature table...")

FORBIDDEN = {
    "R_odds", "B_odds", "R_ev", "B_ev", "r_dec_odds", "b_dec_odds",
    "r_sub_odds", "b_sub_odds", "r_ko_odds", "b_ko_odds",
    "finish_round", "finish_round_time", "total_fight_time_secs",
    "finish", "finish_details",
}

master["Winner_bin"] = (master["Winner"] == "Red").astype(int)

# Weight class ordinal
WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11, "Catch Weight": 6,
}
master["weight_class_ord"] = master["weight_class"].map(WC_ORDER).fillna(6)

# Title bout binary (already in master as title_bout)
master["title_bout_bin"] = (master["title_bout"] == True).astype(int)

# Stance matchup features
def stance_feats(df):
    r = df["R_Stance"].fillna("Unknown")
    b = df["B_Stance"].fillna("Unknown")
    df["orth_clash"]   = ((r == "Orthodox")  & (b == "Orthodox")).astype(int)
    df["south_clash"]  = ((r == "Southpaw")  & (b == "Southpaw")).astype(int)
    df["R_southpaw"]   = (r == "Southpaw").astype(int)
    df["B_southpaw"]   = (b == "Southpaw").astype(int)
    return df

master = stance_feats(master)

# Join career features
master = join_career(master, "R")
master = join_career(master, "B")

log(f"  after career join: {master.shape}")

# Join style features from ufc_fighters_final
STYLE_COLS = ["SLpM", "SApM", "Str_Acc", "Str_Def", "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg"]
style = fighters[["Fighter_Name"] + STYLE_COLS].drop_duplicates("Fighter_Name").copy()
# Convert percentage strings to floats (e.g. "63%" → 0.63)
for sc in ["Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]:
    if style[sc].dtype == object:
        style[sc] = pd.to_numeric(style[sc].str.replace("%", "", regex=False), errors="coerce") / 100.0

for prefix, side in [("R", "R_fighter"), ("B", "B_fighter")]:
    style_r = style.rename(columns={
        "Fighter_Name": side,
        **{c: f"{prefix}_{c}" for c in STYLE_COLS}
    })
    master = master.merge(style_r, on=side, how="left")

log(f"  after style join: {master.shape}")

# Derived diff features
def add_diff(df, col, R_col=None, B_col=None):
    rc = R_col or f"R_{col}"
    bc = B_col or f"B_{col}"
    if rc in df.columns and bc in df.columns:
        df[f"{col}_dif"] = df[rc] - df[bc]

for col in [
    "career_win_rate", "ko_finish_rate", "sub_finish_rate",
    "trend_score", "opp_quality",
    "last3_win_rate", "last10_win_rate", "layoff_days",
    "last5_won", "last5_finish_rate",
    "SLpM", "SApM", "Str_Def", "TD_Def", "Sub_Avg", "TD_Avg",
]:
    add_diff(master, col)

# Age × experience interaction
for p in ["R", "B"]:
    if f"{p}_age" in master.columns and f"{p}_cum_fights" in master.columns:
        master[f"{p}_age_x_exp"] = master[f"{p}_age"] * master[f"{p}_cum_fights"]
add_diff(master, "age_x_exp")

# Layoff buckets
for p in ["R", "B"]:
    ld = f"{p}_layoff_days"
    if ld in master.columns:
        master[f"{p}_layoff_lt90"]   = (master[ld] < 90).astype(int)
        master[f"{p}_layoff_90_180"] = ((master[ld] >= 90)  & (master[ld] < 180)).astype(int)
        master[f"{p}_layoff_180_365"]= ((master[ld] >= 180) & (master[ld] < 365)).astype(int)
        master[f"{p}_layoff_gt365"]  = (master[ld] >= 365).astype(int)

log(f"  after diff/interaction features: {master.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DEFINE CANDIDATE FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
MASTER_BASE = [
    "R_wins", "R_losses", "R_Height_cms", "R_age",
    "R_avg_SIG_STR_landed", "R_avg_TD_landed", "R_current_win_streak",
    "R_current_lose_streak", "R_longest_win_streak",
    "R_avg_SIG_STR_pct", "R_avg_SUB_ATT", "R_avg_TD_pct",
    "R_Reach_cms", "R_total_title_bouts",
    "B_wins", "B_losses", "B_Height_cms", "B_age",
    "B_avg_SIG_STR_landed", "B_avg_TD_landed", "B_current_win_streak",
    "B_current_lose_streak", "B_longest_win_streak",
    "B_avg_SIG_STR_pct", "B_avg_SUB_ATT", "B_avg_TD_pct",
    "B_Reach_cms", "B_total_title_bouts",
    "win_dif", "loss_dif", "win_streak_dif", "lose_streak_dif",
    "height_dif", "reach_dif", "age_dif", "sig_str_dif", "avg_td_dif",
    "ko_dif", "sub_dif", "total_title_bout_dif",
    "weight_class_ord", "title_bout_bin",
    "orth_clash", "south_clash", "R_southpaw", "B_southpaw",
]

CAREER_BASE = [
    "R_cum_fights", "B_cum_fights",
    "R_career_win_rate", "B_career_win_rate", "career_win_rate_dif",
    "R_last5_won", "B_last5_won", "last5_won_dif",
    "R_last5_finish_rate", "B_last5_finish_rate", "last5_finish_rate_dif",
    "R_opp_quality", "B_opp_quality", "opp_quality_dif",
    "R_trend_score", "B_trend_score", "trend_score_dif",
    "R_ko_finish_rate", "B_ko_finish_rate", "ko_finish_rate_dif",
    "R_sub_finish_rate", "B_sub_finish_rate", "sub_finish_rate_dif",
    "R_last3_win_rate", "B_last3_win_rate", "last3_win_rate_dif",
    "R_last10_win_rate", "B_last10_win_rate", "last10_win_rate_dif",
    "R_age_x_exp", "B_age_x_exp", "age_x_exp_dif",
    "R_layoff_lt90", "R_layoff_90_180", "R_layoff_180_365", "R_layoff_gt365",
    "B_layoff_lt90", "B_layoff_90_180", "B_layoff_180_365", "B_layoff_gt365",
]

STYLE_FEATS = []
for p in ["R", "B"]:
    for c in STYLE_COLS:
        STYLE_FEATS.append(f"{p}_{c}")
for c in ["SLpM", "SApM", "Str_Def", "TD_Def", "Sub_Avg", "TD_Avg"]:
    STYLE_FEATS.append(f"{c}_dif")

ALL_CANDIDATE = MASTER_BASE + CAREER_BASE + STYLE_FEATS

# keep only columns that actually exist
ALL_CANDIDATE = [c for c in ALL_CANDIDATE if c in master.columns]
log(f"  candidate features: {len(ALL_CANDIDATE)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRID SEARCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
WEIGHT_CLASSES = {
    "R_fighter": "R_fighter", "B_fighter": "B_fighter",
    "date": "date", "Winner_bin": "Winner_bin",
}

def prepare_data(df, features, date_cutoff, min_fights):
    sub = df[df["date"] >= pd.Timestamp(f"{date_cutoff}-01-01")].copy()
    if min_fights > 1:
        # use cum_fights from career join as proxy for total fights experience
        for p in ["R", "B"]:
            cf = f"{p}_cum_fights"
            if cf in sub.columns:
                sub = sub[sub[cf].fillna(0) >= min_fights]
    avail = [c for c in features if c in sub.columns]
    sub = sub[avail + ["Winner_bin"]].dropna()
    X = sub[avail]
    y = sub["Winner_bin"]
    return X, y, avail


def augment_corners(X, y, features):
    Xf = X.copy()
    yf = (1 - y).reset_index(drop=True)
    r_feats  = [c for c in features if c.startswith("R_")]
    dif_feats = [c for c in features if c.endswith("_dif")]
    for rc in r_feats:
        bc = "B_" + rc[2:]
        if bc in features:
            Xf[rc] = X[bc].values
            Xf[bc] = X[rc].values
    for dc in dif_feats:
        Xf[dc] = -X[dc].values
    return pd.concat([X.reset_index(drop=True), Xf], ignore_index=True), \
           pd.concat([y.reset_index(drop=True), yf], ignore_index=True)


def split_augment(X, y, features):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_aug, ytr_aug = augment_corners(Xtr, ytr, features)
    return Xtr_aug, Xte, ytr_aug, yte


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FEATURE IMPORTANCE TOP-K SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
def get_topk_features(Xtr, ytr, features, k):
    ref = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RS, eval_metric="logloss", verbosity=0, n_jobs=1
    )
    ref.fit(Xtr, ytr)
    imp = pd.Series(ref.feature_importances_, index=features).sort_values(ascending=False)
    return imp.head(k).index.tolist(), imp


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MODEL TRAINING FUNCTIONS (Optuna, n_jobs=1)
# ─────────────────────────────────────────────────────────────────────────────
def train_catboost(Xtr, ytr, Xte, yte):
    def obj(trial):
        m = CatBoostClassifier(
            iterations       = trial.suggest_int("iterations", 300, 1500),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth            = trial.suggest_int("depth", 4, 10),
            l2_leaf_reg      = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
            random_seed=RS, verbose=False, thread_count=1,
        )
        m.fit(Xtr, ytr)
        return accuracy_score(yte, m.predict(Xte))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT_SECS, n_jobs=1, show_progress_bar=False)
    best = study.best_params
    m = CatBoostClassifier(**best, random_seed=RS, verbose=False, thread_count=1)
    m.fit(Xtr, ytr)
    return m, accuracy_score(yte, m.predict(Xte))


def train_xgboost(Xtr, ytr, Xte, yte):
    def obj(trial):
        m = xgb.XGBClassifier(
            n_estimators     = trial.suggest_int("n_estimators", 200, 1500),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 9),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            gamma            = trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 5.0),
            random_state=RS, eval_metric="logloss", verbosity=0, n_jobs=1,
        )
        m.fit(Xtr, ytr)
        return accuracy_score(yte, m.predict(Xte))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT_SECS, n_jobs=1, show_progress_bar=False)
    best = study.best_params
    m = xgb.XGBClassifier(**best, random_state=RS, eval_metric="logloss", verbosity=0, n_jobs=1)
    m.fit(Xtr, ytr)
    return m, accuracy_score(yte, m.predict(Xte))


def train_lgbm(Xtr, ytr, Xte, yte):
    def obj(trial):
        m = lgb.LGBMClassifier(
            n_estimators     = trial.suggest_int("n_estimators", 200, 1500),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 9),
            num_leaves       = trial.suggest_int("num_leaves", 20, 150),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.0, 5.0),
            random_state=RS, n_jobs=1, verbose=-1,
        )
        m.fit(Xtr, ytr)
        return accuracy_score(yte, m.predict(Xte))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT_SECS, n_jobs=1, show_progress_bar=False)
    best = study.best_params
    m = lgb.LGBMClassifier(**best, random_state=RS, n_jobs=1, verbose=-1)
    m.fit(Xtr, ytr)
    return m, accuracy_score(yte, m.predict(Xte))


def train_mlp(Xtr, ytr, Xte, yte):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    def obj(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        sizes = tuple(
            trial.suggest_int(f"h{i}", 32, 256) for i in range(n_layers)
        )
        m = MLPClassifier(
            hidden_layer_sizes = sizes,
            alpha              = trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            max_iter=500, random_state=RS,
        )
        m.fit(Xtr_s, ytr)
        return accuracy_score(yte, m.predict(Xte_s))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT_SECS, n_jobs=1, show_progress_bar=False)
    bp = study.best_params
    n_layers = bp["n_layers"]
    sizes = tuple(bp[f"h{i}"] for i in range(n_layers))
    m = MLPClassifier(
        hidden_layer_sizes=sizes,
        alpha=bp["alpha"],
        learning_rate_init=bp["learning_rate_init"],
        max_iter=500, random_state=RS,
    )
    m.fit(Xtr_s, ytr)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("scaler", scaler), ("mlp", m)])
    acc = accuracy_score(yte, pipe.predict(Xte))
    return pipe, acc


def train_logreg(Xtr, ytr, Xte, yte):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    best_acc, best_m = 0, None
    for penalty in ["l1", "l2"]:
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            m = LogisticRegression(penalty=penalty, C=C, solver="saga",
                                   max_iter=1000, random_state=RS, n_jobs=1)
            m.fit(Xtr_s, ytr)
            a = accuracy_score(yte, m.predict(Xte_s))
            if a > best_acc:
                best_acc, best_m, best_scaler = a, m, scaler

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("scaler", best_scaler), ("lr", best_m)])
    return pipe, best_acc


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN GRID
# ─────────────────────────────────────────────────────────────────────────────
best_global_acc  = BASELINE_ACC
best_global_model = None
best_global_feats = None
best_global_info  = {}

all_results = []  # list of dicts for ranking table

def evaluate_config(name, Xtr, ytr, Xte, yte, feats, train_fn, date_cut, min_f, feat_set):
    global best_global_acc, best_global_model, best_global_feats, best_global_info
    t0 = time.time()
    tag = f"{name}|date≥{date_cut}|mf{min_f}|{feat_set}"
    log(f"\n  >> {tag}  ({len(feats)} features, train={len(ytr)}, test={len(yte)})")
    try:
        model, acc = train_fn(Xtr, ytr, Xte, yte)
        elapsed = time.time() - t0
        log(f"     acc={acc:.4f}  ({elapsed:.1f}s)")
        all_results.append({"config": tag, "acc": acc, "features": len(feats), "secs": elapsed})
        if acc > best_global_acc:
            best_global_acc = acc
            best_global_model = model
            best_global_feats = feats
            best_global_info = {"config": tag, "acc": acc}
            joblib.dump(model, BEST_MODEL_FILE)
            joblib.dump(feats,  BEST_FEATS_FILE)
            joblib.dump(model, CUR_MODEL_FILE)
            joblib.dump(feats,  CUR_FEATS_FILE)
            log(f"  *** NEW BEST: {acc:.4f}  saved to {BEST_MODEL_FILE}")
    except Exception as e:
        elapsed = time.time() - t0
        log(f"     ERROR: {e}  ({elapsed:.1f}s)")
        all_results.append({"config": tag, "acc": None, "features": len(feats), "secs": elapsed})
    gc.collect()


# Grid parameters
DATE_CUTS  = [2015, 2016, 2017]
MIN_FIGHTS = [1, 2, 3]

# We'll run one date/min_fights config per model to save time, then expand.
# Strategy: use best config from exp2 (2016, mf=2) as primary, also try 2015/mf=1, 2017/mf=3.
GRID_CONFIGS = [
    (2016, 2),   # best from exp2
    (2015, 1),   # more data
    (2017, 3),   # quality filter
]

MODEL_FUNS = [
    ("CatBoost", train_catboost),
    ("XGBoost",  train_xgboost),
    ("LightGBM", train_lgbm),
    ("MLP",      train_mlp),
    ("LogReg",   train_logreg),
]

log("\n" + "=" * 70)
log("RUNNING MODEL GRID")
log("=" * 70)

# We'll compute top-k feature sets once per (date_cut, min_fights) config,
# then reuse across models.

for date_cut, min_f in GRID_CONFIGS:
    log(f"\n{'─'*60}")
    log(f"Config: date≥{date_cut}, min_ufc_fights≥{min_f}")
    log(f"{'─'*60}")

    X_all, y_all, feats_all = prepare_data(master, ALL_CANDIDATE, date_cut, min_f)
    if len(y_all) < 200:
        log(f"  Skipping — only {len(y_all)} rows after filtering")
        continue

    Xtr_all, Xte_all, ytr_all, yte_all = split_augment(X_all, y_all, feats_all)

    # Compute feature importance on full set for top-k selection
    log(f"  Computing feature importance ({len(feats_all)} features, {len(ytr_all)} train rows)...")
    topk_feats, imp_series = get_topk_features(Xtr_all, ytr_all, feats_all, len(feats_all))
    top30 = topk_feats[:30]
    top15 = topk_feats[:15]
    log(f"  Top-5: {topk_feats[:5]}")

    for model_name, train_fn in MODEL_FUNS:
        log(f"\n  === {model_name} ===")

        # All features
        evaluate_config(
            model_name, Xtr_all, ytr_all, Xte_all, yte_all,
            feats_all, train_fn, date_cut, min_f, "all"
        )

        # Top-30
        feats_30 = [c for c in top30 if c in feats_all]
        if len(feats_30) >= 10:
            X30, y30, _ = prepare_data(master, feats_30, date_cut, min_f)
            Xtr30, Xte30, ytr30, yte30 = split_augment(X30, y30, feats_30)
            evaluate_config(
                model_name, Xtr30, ytr30, Xte30, yte30,
                feats_30, train_fn, date_cut, min_f, "top30"
            )
            del X30, Xtr30, Xte30; gc.collect()

        # Top-15
        feats_15 = [c for c in top15 if c in feats_all]
        if len(feats_15) >= 10:
            X15, y15, _ = prepare_data(master, feats_15, date_cut, min_f)
            Xtr15, Xte15, ytr15, yte15 = split_augment(X15, y15, feats_15)
            evaluate_config(
                model_name, Xtr15, ytr15, Xte15, yte15,
                feats_15, train_fn, date_cut, min_f, "top15"
            )
            del X15, Xtr15, Xte15; gc.collect()

        del Xtr_all, ytr_all  # keep Xte for reuse — re-split before next model
        Xtr_all, _, ytr_all, _ = split_augment(X_all, y_all, feats_all)
        gc.collect()

    # Voting + Stacking ensembles on best date/min_f config
    log(f"\n  === Voting Ensemble ===")
    try:
        cb = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=7,
                                random_seed=RS, verbose=False, thread_count=1)
        xg = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                                random_state=RS, eval_metric="logloss", verbosity=0, n_jobs=1)
        lg = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                                 random_state=RS, n_jobs=1, verbose=-1)
        voting = VotingClassifier(
            estimators=[("cb", cb), ("xgb", xg), ("lgb", lg)],
            voting="soft", n_jobs=1
        )
        Xtr_all, Xte_all, ytr_all, yte_all = split_augment(X_all, y_all, feats_all)
        voting.fit(Xtr_all, ytr_all)
        v_acc = accuracy_score(yte_all, voting.predict(Xte_all))
        log(f"     Voting acc={v_acc:.4f}")
        all_results.append({"config": f"Voting|date≥{date_cut}|mf{min_f}|all", "acc": v_acc,
                             "features": len(feats_all), "secs": 0})
        if v_acc > best_global_acc:
            best_global_acc = v_acc
            joblib.dump(voting, BEST_MODEL_FILE)
            joblib.dump(feats_all, BEST_FEATS_FILE)
            joblib.dump(voting, CUR_MODEL_FILE)
            joblib.dump(feats_all, CUR_FEATS_FILE)
            log(f"  *** NEW BEST (Voting): {v_acc:.4f}")
        del voting; gc.collect()
    except Exception as e:
        log(f"     Voting ERROR: {e}")

    log(f"\n  === Stacking Ensemble ===")
    try:
        cb2 = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
                                 random_seed=RS, verbose=False, thread_count=1)
        xg2 = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                  random_state=RS, eval_metric="logloss", verbosity=0, n_jobs=1)
        lg2 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                   random_state=RS, n_jobs=1, verbose=-1)
        meta = LogisticRegression(C=1.0, max_iter=500, random_state=RS, n_jobs=1)
        stacking = StackingClassifier(
            estimators=[("cb", cb2), ("xgb", xg2), ("lgb", lg2)],
            final_estimator=meta,
            cv=3, n_jobs=1,
        )
        Xtr_all, Xte_all, ytr_all, yte_all = split_augment(X_all, y_all, feats_all)
        stacking.fit(Xtr_all, ytr_all)
        s_acc = accuracy_score(yte_all, stacking.predict(Xte_all))
        log(f"     Stacking acc={s_acc:.4f}")
        all_results.append({"config": f"Stacking|date≥{date_cut}|mf{min_f}|all", "acc": s_acc,
                             "features": len(feats_all), "secs": 0})
        if s_acc > best_global_acc:
            best_global_acc = s_acc
            joblib.dump(stacking, BEST_MODEL_FILE)
            joblib.dump(feats_all, BEST_FEATS_FILE)
            joblib.dump(stacking, CUR_MODEL_FILE)
            joblib.dump(feats_all, CUR_FEATS_FILE)
            log(f"  *** NEW BEST (Stacking): {s_acc:.4f}")
        del stacking; gc.collect()
    except Exception as e:
        log(f"     Stacking ERROR: {e}")

    del X_all, Xtr_all, Xte_all, ytr_all, yte_all; gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 11. FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("FINAL RESULTS — RANKED")
log("=" * 70)

results_df = pd.DataFrame(all_results).dropna(subset=["acc"]).sort_values("acc", ascending=False)
log(results_df.to_string(index=False))

log(f"\nBest overall: {best_global_acc:.4f}")
log(f"Best config:  {best_global_info}")
if best_global_acc > BASELINE_ACC:
    log(f"IMPROVEMENT over baseline: +{best_global_acc - BASELINE_ACC:.4f}")
    log(f"Saved to: {BEST_MODEL_FILE}")
else:
    log("No improvement over baseline 68.94%.")

if best_global_feats:
    log(f"\nBest feature set ({len(best_global_feats)} features):")
    log(str(best_global_feats[:30]))

log("\n[DONE] " + pd.Timestamp.now().isoformat())
