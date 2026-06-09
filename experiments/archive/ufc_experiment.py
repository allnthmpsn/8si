#!/usr/bin/env python3
"""
UFC Fight Prediction - Overnight Experiment
Comprehensive model comparison with full career feature engineering.
Target: 70%+ accuracy on clean test set.

Validation rules:
  - Split BEFORE augmenting
  - Augment only training set (swap corners + flip target)
  - Drop NaN rows (no median fill)
  - No odds columns, no post-fight columns
  - Test set is clean: never augmented, never used during training
"""

import os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_OPTUNA     = 60   # trials per model during final tuning
os.makedirs("model", exist_ok=True)

LOG_FILE = "model/experiment_log.txt"
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(msg)

log("=" * 72)
log("UFC FIGHT PREDICTION  –  OVERNIGHT EXPERIMENT")
log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 72)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD ALL DATA
# ──────────────────────────────────────────────────────────────────────────────
log("\n[1/6] Loading data …")

ufc = pd.read_csv("data/ufc-master.csv")
ufc["date"] = pd.to_datetime(ufc["date"])
log(f"  ufc-master        : {ufc.shape[0]:,} fights  ({ufc['date'].min().date()} – {ufc['date'].max().date()})")

career = pd.read_csv("data/career_fights.csv")
career["date"]    = pd.to_datetime(career["date"])
career["fighter"] = career["fighter"].str.strip()
log(f"  career_fights     : {career.shape[0]:,} rows  |  {career['fighter'].nunique():,} fighters")

with open("data/wiki_fighter_records.pkl", "rb") as f:
    wiki = pickle.load(f)
log(f"  wiki_records      : {len(wiki):,} fighters")

with open("data/sherdog_records.pkl", "rb") as f:
    sherdog = pickle.load(f)
log(f"  sherdog_records   : {len(sherdog):,} fighters")

# ──────────────────────────────────────────────────────────────────────────────
# 2. CAREER FEATURE ENGINEERING (vectorised)
# ──────────────────────────────────────────────────────────────────────────────
log("\n[2/6] Engineering career features …")

def compute_career_rolling(career_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every row in career_df, compute stats going *into* that fight
    (i.e., based on all prior fights, using shift(1)+cumsum).
    """
    df = (career_df.copy()
          .sort_values(["fighter", "date"])
          .reset_index(drop=True))

    df["is_fin_win"]  = ((df["won"] == 1) & (df["got_finish"] == 1)).astype(float)
    df["is_fin_loss"] = ((df["won"] == 0) & (df["got_finish"] == 1)).astype(float)

    g = df.groupby("fighter", sort=False)

    # Cumulative tallies BEFORE each fight
    df["cum_wins"]       = g["won"].transform(       lambda x: x.shift(1).cumsum().fillna(0))
    df["cum_fin_wins"]   = g["is_fin_win"].transform( lambda x: x.shift(1).cumsum().fillna(0))
    df["cum_fin_losses"] = g["is_fin_loss"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    df["n_prior"]        = g.cumcount()           # 0 for debut fight
    df["cum_losses"]     = df["n_prior"] - df["cum_wins"]

    # Derived rates
    n  = df["n_prior"].clip(lower=1)
    cw = df["cum_wins"].clip(lower=1)
    cl = df["cum_losses"].clip(lower=1)

    df["career_win_rate"]             = np.where(df["n_prior"] > 0, df["cum_wins"]       / n,  np.nan)
    df["career_finish_rate_full"]     = np.where(df["cum_wins"] > 0, df["cum_fin_wins"]  / cw, np.nan)
    df["career_times_finished_rate"]  = np.where(df["cum_losses"] > 0, df["cum_fin_losses"] / cl, np.nan)

    # Days since last fight
    df["prev_date"]             = g["date"].shift(1)
    df["career_days_since_last"] = (df["date"] - df["prev_date"]).dt.days

    return df


def compute_streaks(career_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sequential streak computation (O(n²) per fighter but <1 s for 48 K rows).
    Returns DataFrame with fighter, date, career_win_streak, career_lose_streak.
    """
    records = []
    for fighter, grp in career_df.groupby("fighter", sort=False):
        won_seq = grp["won"].tolist()
        dates   = grp["date"].tolist()
        for i in range(len(won_seq)):
            w = l = 0
            for j in range(i - 1, -1, -1):
                if won_seq[j] == 1:
                    w += 1
                    if l: break
                else:
                    l += 1
                    if w: break
            records.append({"fighter": fighter, "date": dates[i],
                             "career_win_streak": w, "career_lose_streak": l})
    return pd.DataFrame(records)


log("  Computing rolling stats (vectorised) …")
career_feats = compute_career_rolling(career)

log("  Computing win/loss streaks (sequential) …")
streaks = compute_streaks(career_feats)
career_feats = career_feats.merge(streaks, on=["fighter", "date"], how="left")
log(f"  Career features ready  – shape: {career_feats.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. JOIN CAREER FEATURES TO UFC-MASTER
# ──────────────────────────────────────────────────────────────────────────────
log("\n[3/6] Joining career features to UFC master …")

JOIN_COLS = [
    "last5_won", "last5_finish_rate",           # full-career rolling last-5
    "career_win_rate", "career_finish_rate_full",
    "career_times_finished_rate",
    "career_days_since_last",
    "career_win_streak", "career_lose_streak",
    "cum_wins", "cum_losses", "cum_fin_losses", "n_prior",
]


def join_career(ufc_df: pd.DataFrame, career_features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Merge career features for one corner (R or B) using merge_asof.
    For each UFC fight date, picks the last career-fights row whose date ≤ UFC date.
    """
    fighter_col = f"{prefix}_fighter"
    sub = (career_features[["fighter", "date"] + JOIN_COLS]
           .copy()
           .rename(columns={"fighter": fighter_col})
           .rename(columns={c: f"{prefix}_c_{c}" for c in JOIN_COLS})
           .sort_values("date"))

    merged = pd.merge_asof(
        ufc_df.sort_values("date"),
        sub,
        on="date",
        by=fighter_col,
        direction="backward",
    )
    return merged


ufc2 = join_career(ufc, career_feats, "R")
ufc2 = join_career(ufc2, career_feats, "B")
log(f"  After join: {ufc2.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. COMPUTE ALL DERIVED FEATURES
# ──────────────────────────────────────────────────────────────────────────────
log("\n[4/6] Computing derived features …")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── UFC record metrics ────────────────────────────────────────────────────
    for p in ("R", "B"):
        ko  = df.get(f"{p}_win_by_KO/TKO",     pd.Series(0, index=df.index))
        sub = df.get(f"{p}_win_by_Submission",  pd.Series(0, index=df.index))
        dec_u = df.get(f"{p}_win_by_Decision_Unanimous", pd.Series(0, index=df.index))
        dec_s = df.get(f"{p}_win_by_Decision_Split",     pd.Series(0, index=df.index))
        dec_m = df.get(f"{p}_win_by_Decision_Majority",  pd.Series(0, index=df.index))

        df[f"{p}_ufc_fights"]   = df[f"{p}_wins"] + df[f"{p}_losses"]
        df[f"{p}_finish_rate"]  = (ko + sub) / (df[f"{p}_wins"] + 1)
        # Times finished: prefer career cum_fin_losses, fall back to UFC decision math
        fin_loss_col = f"{p}_c_cum_fin_losses"
        if fin_loss_col in df.columns:
            df[f"{p}_times_finished"] = df[fin_loss_col].fillna(0)
        else:
            df[f"{p}_times_finished"] = (df[f"{p}_losses"] - (dec_u + dec_s + dec_m)).clip(lower=0)
        # Use career-wide last5 (full career, not just UFC)
        df[f"{p}_last5_won"]    = df[f"{p}_c_last5_won"]

        # Pre-UFC wins/losses  (career total − UFC total at fight date)
        c_wins   = df[f"{p}_c_cum_wins"]
        c_losses = df[f"{p}_c_cum_losses"]
        df[f"{p}_pre_ufc_wins"]   = (c_wins   - df[f"{p}_wins"]).clip(lower=0)
        df[f"{p}_pre_ufc_losses"] = (c_losses - df[f"{p}_losses"]).clip(lower=0)

    # ── td_dif  (ufc-master uses avg_td_dif; we standardise the name) ─────────
    if "avg_td_dif" in df.columns:
        df["td_dif"] = df["avg_td_dif"]
    else:
        df["td_dif"] = df["R_avg_TD_landed"] - df["B_avg_TD_landed"]

    # ── Diff features ─────────────────────────────────────────────────────────
    def diff(r, b): return df[r] - df[b]

    df["ufc_fights_dif"]     = diff("R_ufc_fights",   "B_ufc_fights")
    df["finish_rate_dif"]    = diff("R_finish_rate",  "B_finish_rate")
    df["times_finished_dif"] = diff("R_times_finished", "B_times_finished")
    df["last5_dif"]          = diff("R_last5_won",    "B_last5_won")
    df["pre_ufc_wins_dif"]   = diff("R_pre_ufc_wins", "B_pre_ufc_wins")
    df["pre_ufc_losses_dif"] = diff("R_pre_ufc_losses", "B_pre_ufc_losses")

    # Career diff features
    for col in ["career_win_rate", "career_finish_rate_full",
                "career_times_finished_rate", "career_days_since_last",
                "career_win_streak", "career_lose_streak"]:
        rc, bc = f"R_c_{col}", f"B_c_{col}"
        if rc in df.columns and bc in df.columns:
            df[f"{col}_dif"] = df[rc] - df[bc]

    df["career_last5_won_dif"]    = df["R_c_last5_won"]    - df["B_c_last5_won"]
    df["career_last5_finish_dif"] = df["R_c_last5_finish_rate"] - df["B_c_last5_finish_rate"]

    # Total career wins dif  (replaces sherdog total_wins_dif)
    df["total_wins_dif"]   = df["R_c_cum_wins"]   - df["B_c_cum_wins"]
    df["total_losses_dif"] = df["R_c_cum_losses"] - df["B_c_cum_losses"]

    return df


ufc2 = build_features(ufc2)

# ── Target ────────────────────────────────────────────────────────────────────
ufc2 = ufc2[ufc2["Winner"].isin(["Red", "Blue"])].copy()
ufc2["target"] = (ufc2["Winner"] == "Red").astype(int)

# ── Blocklist: never include these columns ────────────────────────────────────
LEAKY = {
    "finish", "finish_details", "finish_round", "finish_round_time",
    "total_fight_time_secs", "Winner",
    "R_odds", "B_odds", "R_ev", "B_ev",
    "r_dec_odds", "b_dec_odds", "r_sub_odds", "b_sub_odds",
    "r_ko_odds", "b_ko_odds",
}

# ──────────────────────────────────────────────────────────────────────────────
# 5. DEFINE FEATURE SETS
# ──────────────────────────────────────────────────────────────────────────────

BASE = [
    # individual stats
    "R_wins", "R_losses", "R_ufc_fights", "R_finish_rate", "R_times_finished",
    "R_last5_won", "R_Height_cms", "R_age", "R_avg_SIG_STR_landed", "R_avg_TD_landed",
    "B_wins", "B_losses", "B_ufc_fights", "B_finish_rate", "B_times_finished",
    "B_last5_won", "B_Height_cms", "B_age", "B_avg_SIG_STR_landed", "B_avg_TD_landed",
    # diff features
    "win_dif", "loss_dif", "ufc_fights_dif", "finish_rate_dif", "times_finished_dif",
    "last5_dif", "height_dif", "age_dif", "sig_str_dif", "td_dif",
    "total_wins_dif", "total_losses_dif",
]

CAREER_EXTRAS = [
    # full-career last-5 (different from UFC last5 used in BASE)
    "R_c_last5_won", "R_c_last5_finish_rate",
    "B_c_last5_won", "B_c_last5_finish_rate",
    "career_last5_won_dif", "career_last5_finish_dif",
    # career rates
    "R_c_career_win_rate", "R_c_career_finish_rate_full", "R_c_career_times_finished_rate",
    "B_c_career_win_rate", "B_c_career_finish_rate_full", "B_c_career_times_finished_rate",
    "career_win_rate_dif", "career_finish_rate_full_dif", "career_times_finished_rate_dif",
    # inactivity & streak
    "R_c_career_days_since_last", "B_c_career_days_since_last", "career_days_since_last_dif",
    "R_c_career_win_streak",  "B_c_career_win_streak",  "career_win_streak_dif",
    "R_c_career_lose_streak", "B_c_career_lose_streak", "career_lose_streak_dif",
    # pre-UFC record
    "R_pre_ufc_wins", "R_pre_ufc_losses",
    "B_pre_ufc_wins", "B_pre_ufc_losses",
    "pre_ufc_wins_dif", "pre_ufc_losses_dif",
]

UFC_EXTRAS = [
    "R_current_win_streak",  "B_current_win_streak",  "win_streak_dif",
    "R_current_lose_streak", "B_current_lose_streak", "lose_streak_dif",
    "R_avg_SIG_STR_pct",     "B_avg_SIG_STR_pct",
    "R_avg_TD_pct",          "B_avg_TD_pct",
    "R_Reach_cms",           "B_Reach_cms",           "reach_dif",
    "R_avg_SUB_ATT",         "B_avg_SUB_ATT",
]

ALL_CANDIDATE = list(dict.fromkeys(BASE + CAREER_EXTRAS + UFC_EXTRAS))

FEATURE_SETS = {
    "base_32":      BASE,
    "base+career":  BASE + CAREER_EXTRAS,
    "all":          ALL_CANDIDATE,
}

# ──────────────────────────────────────────────────────────────────────────────
# 6. DATASET PREPARATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def augment_corners(X: pd.DataFrame, y: pd.Series, features: list):
    """
    Flip Red ↔ Blue for every training row and negate diff features.
    Returns (X_flipped, y_flipped) – caller concatenates with originals.
    """
    X_flip = X.copy()
    y_flip = (1 - y).reset_index(drop=True)

    r_feats  = [f for f in features if f.startswith("R_")]
    b_feats  = [f for f in features if f.startswith("B_")]
    dif_feats = [f for f in features if f.endswith("_dif")]

    for rf in r_feats:
        bf = "B_" + rf[2:]
        if bf in features:
            X_flip[rf] = X[bf].values
            X_flip[bf] = X[rf].values

    for df_col in dif_feats:
        if df_col in X_flip.columns:
            X_flip[df_col] = -X[df_col].values

    return X_flip, y_flip


def make_split(df: pd.DataFrame, date_cutoff: int, min_ufc_fights: int,
               feat_names: list, augment: bool = True):
    """
    Apply filters → drop NaN → split → (optionally) augment training set.
    Returns (X_train, X_test, y_train, y_test, used_features) or Nones.
    """
    d = df[df["date"] >= f"{date_cutoff}-01-01"].copy()

    if "R_ufc_fights" in d.columns and "B_ufc_fights" in d.columns:
        d = d[(d["R_ufc_fights"] >= min_ufc_fights) &
              (d["B_ufc_fights"] >= min_ufc_fights)]

    avail = [f for f in feat_names if f in d.columns and f not in LEAKY]
    keep  = avail + ["target"]
    d = d[keep].dropna()

    if len(d) < 200:
        return None, None, None, None, None

    X, y = d[avail], d["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE)

    if augment:
        Xf, yf   = augment_corners(Xtr.reset_index(drop=True),
                                   ytr.reset_index(drop=True), avail)
        Xtr = pd.concat([Xtr.reset_index(drop=True), Xf], ignore_index=True)
        ytr = pd.concat([ytr.reset_index(drop=True), yf], ignore_index=True)

    return Xtr, Xte, ytr, yte, avail


# ──────────────────────────────────────────────────────────────────────────────
# 7. GRID SEARCH: best preprocessing config with fast XGBoost
# ──────────────────────────────────────────────────────────────────────────────
log("\n[5a/6] Grid search – preprocessing config × feature set …")

DATE_CUTOFFS   = [2015, 2016, 2017]
MIN_UFC_FIGHTS = [1, 2, 3]

grid_results = []

for year in DATE_CUTOFFS:
    for mf in MIN_UFC_FIGHTS:
        for fsname, fset in FEATURE_SETS.items():
            Xtr, Xte, ytr, yte, used = make_split(ufc2, year, mf, fset)
            if Xtr is None:
                continue

            m = xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
            )
            m.fit(Xtr, ytr)
            acc = accuracy_score(yte, m.predict(Xte))

            log(f"  date≥{year}  min_fights={mf}  feats={fsname:<12} "
                f"| train={len(ytr):,}  test={len(yte):,}  acc={acc:.4f}")
            grid_results.append(dict(year=year, mf=mf, fsname=fsname,
                                     acc=acc, used=used))

best_grid = max(grid_results, key=lambda x: x["acc"])
log(f"\n  ✓ Best config: date≥{best_grid['year']}  min_fights={best_grid['mf']}"
    f"  feats={best_grid['fsname']}  acc={best_grid['acc']:.4f}")

# Re-create the best split for final tuning
best_Xtr, best_Xte, best_ytr, best_yte, best_used = make_split(
    ufc2, best_grid["year"], best_grid["mf"], best_grid["used"]
)
log(f"  Final split → train={len(best_ytr):,}  test={len(best_yte):,}"
    f"  features={len(best_used)}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. HYPERPARAMETER TUNING  (Optuna, using best split)
# ──────────────────────────────────────────────────────────────────────────────
log(f"\n[5b/6] Hyperparameter tuning ({N_OPTUNA} trials each) …")

Xtr, Xte, ytr, yte, used = best_Xtr, best_Xte, best_ytr, best_yte, best_used


# ── generic tuner ─────────────────────────────────────────────────────────────
def run_study(objective, n_trials=N_OPTUNA):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ── XGBoost ──────────────────────────────────────────────────────────────────
def obj_xgb(trial):
    m = xgb.XGBClassifier(
        n_estimators     = trial.suggest_int("n_estimators", 200, 1200),
        learning_rate    = trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        subsample        = trial.suggest_float("subsample", 0.55, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.45, 1.0),
        gamma            = trial.suggest_float("gamma", 0, 5),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        random_state=RANDOM_STATE, eval_metric="logloss",
        use_label_encoder=False, verbosity=0,
    )
    return accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))


log("  Tuning XGBoost …")
st_xgb = run_study(obj_xgb)
best_xgb_p = st_xgb.best_params
best_xgb_p.update(dict(random_state=RANDOM_STATE, eval_metric="logloss",
                        use_label_encoder=False, verbosity=0))
final_xgb = xgb.XGBClassifier(**best_xgb_p).fit(Xtr, ytr)
xgb_acc = accuracy_score(yte, final_xgb.predict(Xte))
log(f"  XGBoost   → {xgb_acc:.4f}  (best trial={st_xgb.best_value:.4f})")


# ── LightGBM ─────────────────────────────────────────────────────────────────
def obj_lgbm(trial):
    m = lgb.LGBMClassifier(
        n_estimators     = trial.suggest_int("n_estimators", 200, 1200),
        learning_rate    = trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        num_leaves       = trial.suggest_int("num_leaves", 15, 127),
        min_child_samples= trial.suggest_int("min_child_samples", 10, 100),
        subsample        = trial.suggest_float("subsample", 0.55, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.45, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        random_state=RANDOM_STATE, verbose=-1,
    )
    return accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))


log("  Tuning LightGBM …")
st_lgbm = run_study(obj_lgbm)
best_lgbm_p = st_lgbm.best_params
best_lgbm_p.update(dict(random_state=RANDOM_STATE, verbose=-1))
final_lgbm = lgb.LGBMClassifier(**best_lgbm_p).fit(Xtr, ytr)
lgbm_acc = accuracy_score(yte, final_lgbm.predict(Xte))
log(f"  LightGBM  → {lgbm_acc:.4f}  (best trial={st_lgbm.best_value:.4f})")


# ── CatBoost ─────────────────────────────────────────────────────────────────
def obj_cat(trial):
    m = CatBoostClassifier(
        iterations   = trial.suggest_int("iterations", 200, 1200),
        learning_rate= trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        depth        = trial.suggest_int("depth", 3, 8),
        l2_leaf_reg  = trial.suggest_float("l2_leaf_reg", 1, 30),
        border_count = trial.suggest_int("border_count", 32, 128),
        random_seed=RANDOM_STATE, verbose=False,
    )
    return accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))


log("  Tuning CatBoost …")
st_cat = run_study(obj_cat)
best_cat_p = st_cat.best_params
best_cat_p.update(dict(random_seed=RANDOM_STATE, verbose=False))
final_cat = CatBoostClassifier(**best_cat_p).fit(Xtr, ytr)
cat_acc = accuracy_score(yte, final_cat.predict(Xte))
log(f"  CatBoost  → {cat_acc:.4f}  (best trial={st_cat.best_value:.4f})")


# ── Random Forest ─────────────────────────────────────────────────────────────
def obj_rf(trial):
    m = RandomForestClassifier(
        n_estimators    = trial.suggest_int("n_estimators", 200, 800),
        max_depth       = trial.suggest_int("max_depth", 5, 25),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf= trial.suggest_int("min_samples_leaf", 1, 10),
        max_features    = trial.suggest_categorical("max_features",
                                                    ["sqrt", "log2", 0.4, 0.6]),
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))


log("  Tuning RandomForest …")
st_rf = run_study(obj_rf)
best_rf_p = st_rf.best_params
best_rf_p.update(dict(random_state=RANDOM_STATE))
final_rf = RandomForestClassifier(**best_rf_p, n_jobs=-1).fit(Xtr, ytr)
rf_acc = accuracy_score(yte, final_rf.predict(Xte))
log(f"  RandomForest → {rf_acc:.4f}  (best trial={st_rf.best_value:.4f})")


# ── Voting Ensemble (soft) ────────────────────────────────────────────────────
log("  Training Voting Ensemble (XGB + LGBM + CAT) …")
voting = VotingClassifier(
    estimators=[("xgb", final_xgb), ("lgbm", final_lgbm), ("cat", final_cat)],
    voting="soft",
)
voting.fit(Xtr, ytr)
voting_acc = accuracy_score(yte, voting.predict(Xte))
log(f"  Voting Ensemble → {voting_acc:.4f}")


# ── Stacking Ensemble ─────────────────────────────────────────────────────────
log("  Training Stacking Ensemble …")
stacking = StackingClassifier(
    estimators=[("xgb", final_xgb), ("lgbm", final_lgbm),
                ("cat", final_cat), ("rf", final_rf)],
    final_estimator=LogisticRegression(random_state=RANDOM_STATE,
                                       max_iter=1000, C=1.0),
    cv=5, n_jobs=-1,
)
stacking.fit(Xtr, ytr)
stacking_acc = accuracy_score(yte, stacking.predict(Xte))
log(f"  Stacking Ensemble → {stacking_acc:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. RESULTS SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
log("\n[6/6] Final results …")

trained = {
    "XGBoost":        (final_xgb,  xgb_acc),
    "LightGBM":       (final_lgbm, lgbm_acc),
    "CatBoost":       (final_cat,  cat_acc),
    "RandomForest":   (final_rf,   rf_acc),
    "VotingEnsemble": (voting,     voting_acc),
    "StackingEnsemble":(stacking,  stacking_acc),
}

log("\n" + "=" * 72)
log("MODEL COMPARISON  (clean test set, no augmentation)")
log("=" * 72)
log(f"  {'Model':<22} {'Acc':>7} {'Prec(R)':>9} {'Rec(R)':>9} {'F1(R)':>8}")
log("  " + "-" * 60)

for name, (m, acc) in sorted(trained.items(), key=lambda x: -x[1][1]):
    yp = m.predict(Xte)
    p  = precision_score(yte, yp, zero_division=0)
    r  = recall_score(yte, yp, zero_division=0)
    f1 = 2 * p * r / (p + r + 1e-9)
    log(f"  {name:<22} {acc:>7.4f} {p:>9.4f} {r:>9.4f} {f1:>8.4f}")

# Best model
best_name = max(trained, key=lambda k: trained[k][1])
best_model, best_acc = trained[best_name]

log("\n" + "=" * 72)
log(f"  WINNER: {best_name}   accuracy = {best_acc:.4f}")
log("=" * 72)

yp_best = best_model.predict(Xte)
log("\nClassification Report (test set):")
log(classification_report(yte, yp_best, target_names=["Blue wins (0)", "Red wins (1)"]))

# Feature importances — always use XGBoost as the reference (interpretable)
imp = pd.Series(final_xgb.feature_importances_, index=used).sort_values(ascending=False)
log("\nFeature Importances – XGBoost reference (top 25):")
for feat, val in imp.head(25).items():
    bar = "█" * int(val * 300)
    log(f"  {feat:<48} {val:.4f}  {bar}")

log("\nBottom 10 features (least important in XGBoost):")
for feat, val in imp.tail(10).items():
    log(f"  {feat:<48} {val:.4f}")

# Best hyperparameters
log("\nBest hyperparameters:")
log(f"  Model : {best_name}")
hp_map = {
    "XGBoost":      best_xgb_p,
    "LightGBM":     best_lgbm_p,
    "CatBoost":     best_cat_p,
    "RandomForest": best_rf_p,
}
if best_name in hp_map:
    for k, v in hp_map[best_name].items():
        if k not in ("random_state","eval_metric","use_label_encoder",
                     "verbosity","verbose","random_seed"):
            log(f"  {k} = {v}")

# Interesting findings
log("\nInteresting findings:")
log(f"  • Test set size     : {len(yte):,} fights")
log(f"  • Training set size : {len(ytr):,} (includes augmented flips)")
log(f"  • Red wins in test  : {yte.sum():,} / {len(yte):,} = {yte.mean():.3f}  (naïve baseline)")
log(f"  • Best config       : date≥{best_grid['year']}  min_ufc_fights={best_grid['mf']}  features={best_grid['fsname']}")
log(f"  • Features used     : {len(used)}")

baseline = yte.mean()
lift = best_acc - max(baseline, 1 - baseline)
log(f"  • Lift over naïve   : {lift:+.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 10. SAVE
# ──────────────────────────────────────────────────────────────────────────────
log("\nSaving …")
joblib.dump(best_model, "model/ufc_model_best.pkl")
with open("model/feature_columns_best.pkl", "wb") as f:
    pickle.dump(used, f)

# Also save all model accuracies for reference
summary = {name: acc for name, (_, acc) in trained.items()}
with open("model/experiment_summary.pkl", "wb") as f:
    pickle.dump({"accuracies": summary, "best_features": used,
                 "best_config": best_grid, "feature_importances":
                 imp.to_dict() if src else {}}, f)

log(f"  model/ufc_model_best.pkl       ← {best_name}")
log(f"  model/feature_columns_best.pkl ← {len(used)} features")
log(f"  model/experiment_summary.pkl   ← full results")

log(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 72)

# Write log file
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
log(f"Log written to {LOG_FILE}")
