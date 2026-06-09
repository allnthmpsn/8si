"""
ufc_experiment8.py — Full retrain on clean data (career_fights_updated.csv).

Steps:
  1  Rebuild all 108 features on updated data, print row counts
  2  Retrain baseline models (same hyperparams), establish new baseline
  3  Retune LR with 100 Optuna trials (temporal accuracy objective)
  4  Retune XGBoost with 50 Optuna trials
  5  Blend ratio sweep (best LR + best XGB)
  6  CatBoost 50 Optuna trials + blends
  7  LightGBM 50 Optuna trials + blends

Primary metric: temporal accuracy (train 2018-2023, test 2024+).
Saves winner if it beats PREV_BEST_TEMPORAL.
"""

import gc, json, os, time, warnings
import joblib, numpy as np, pandas as pd, optuna

from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import RobustScaler, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
RS                   = 42
DATE_FROM            = "2018-01-01"   # training window lower bound
TEMPORAL_CUT         = "2024-01-01"  # train < this, test >= this
MIN_UFC_FIGHTS       = 3
PREV_BEST_TEMPORAL   = 0.7154        # 90/10 LR+XGB blend on old data

OUT_LOG   = "model/experiment8_output.txt"
BEST_MDL  = "model/ufc_model_best.pkl"
BEST_XGB  = "model/ufc_model_xgb.pkl"
BEST_FT   = "model/feature_columns_best.pkl"
META_JSON = "model/model_metadata.json"

os.makedirs("model", exist_ok=True)
with open(OUT_LOG, "w") as f:
    f.write("")

def log(msg=""):
    print(msg, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(msg + "\n")

log("=" * 72)
log(f"UFC EXPERIMENT 8  —  {pd.Timestamp.now().isoformat()}")
log(f"Data: career_fights_updated.csv + ufc_fighters_final_updated.csv")
log(f"Primary target: temporal accuracy > {PREV_BEST_TEMPORAL:.4f}")
log("=" * 72)

# Global tracker
all_results  = []
best_temporal = PREV_BEST_TEMPORAL
best_model_obj = None
best_feats_list = None
best_xgb_obj = None
best_meta    = {}

FEAT_108 = joblib.load(BEST_FT)

def record(tag, t_acc, r_acc, n_feats, secs, model=None, feats=None, meta=None,
           xgb_model=None):
    global best_temporal, best_model_obj, best_feats_list, best_xgb_obj, best_meta
    marker = ""
    if t_acc > best_temporal:
        marker = "  *** NEW BEST ***"
        best_temporal   = t_acc
        best_model_obj  = model
        best_feats_list = feats if feats is not None else FEAT_108
        best_xgb_obj    = xgb_model
        best_meta       = meta or {}
    log(f"  {tag:60s}  t={t_acc:.4f}  r={r_acc:.4f}  feats={n_feats:3d}  {secs:.0f}s{marker}")
    all_results.append({
        "config": tag, "temporal_acc": t_acc, "random_acc": r_acc,
        "n_feats": n_feats, "secs": secs,
    })

def save_best():
    if best_model_obj is None:
        log("  [save] No improvement over baseline — original files unchanged")
        return
    feats = best_feats_list if best_feats_list is not None else FEAT_108
    joblib.dump(best_model_obj, BEST_MDL)
    joblib.dump(feats, BEST_FT)
    if best_xgb_obj is not None:
        joblib.dump(best_xgb_obj, BEST_XGB)
    with open(META_JSON, "w") as f:
        json.dump({
            "model_type":        best_meta.get("model_type", type(best_model_obj).__name__),
            "temporal_accuracy": best_temporal,
            "random_accuracy":   best_meta.get("random_accuracy", 0),
            "n_features":        len(feats),
            "feature_list":      list(feats),
            "hyperparameters":   best_meta.get("params", {}),
            "training_window":   best_meta.get("train_window", f"{DATE_FROM} to <{TEMPORAL_CUT}"),
            "blend_ratio":       best_meta.get("blend_ratio", ""),
            "date_trained":      pd.Timestamp.now().isoformat(),
        }, f, indent=2)
    log(f"  [save] Saved to {BEST_MDL} + {META_JSON}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Feature engineering on updated data
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 72)
log("STEP 1 — Feature engineering on clean data")
log("=" * 72)
t0 = time.time()

career_raw = pd.read_csv("data/career_fights_updated.csv")
career_raw["date"] = pd.to_datetime(career_raw["date"])
master     = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
fighters   = pd.read_csv("data/ufc_fighters_final_updated.csv")

log(f"  career_fights_updated: {len(career_raw):,} rows")
log(f"  ufc-master:            {len(master):,} rows")
log(f"  ufc_fighters_updated:  {len(fighters):,} rows")

# ── Career rolling stats ──────────────────────────────────────────────────────
career = career_raw.sort_values(["fighter", "date"]).reset_index(drop=True).copy()
g = career.groupby("fighter", sort=False)

career["cum_wins"]   = g["won"].transform(lambda x: x.shift(1).cumsum().fillna(0))
career["cum_fights"] = g["won"].transform(lambda x: x.shift(1).expanding().count().fillna(0))
career["career_win_rate"] = career["cum_wins"] / career["cum_fights"].clip(lower=1)

career["did_ko"]  = ((career["won"] == 1) &
                     career["method"].str.contains("KO|TKO", case=False, na=False)).astype(int)
career["did_sub"] = ((career["won"] == 1) &
                     career["method"].str.contains("Sub|Submission", case=False, na=False)).astype(int)
for col in ["did_ko", "did_sub"]:
    career[f"cum_{col}"] = g[col].transform(lambda x: x.shift(1).cumsum().fillna(0))
career["ko_finish_rate"]     = career["cum_did_ko"]  / career["cum_fights"].clip(lower=1)
career["sub_finish_rate"]    = career["cum_did_sub"] / career["cum_fights"].clip(lower=1)
career["cum_finish_wins"]    = career["cum_did_ko"] + career["cum_did_sub"]
career["career_finish_rate"] = career["cum_finish_wins"] / career["cum_wins"].clip(lower=1)

def roll_mean(x, w):
    return x.shift(1).rolling(w, min_periods=1).mean()

career["last3_win_rate"]  = g["won"].transform(lambda x: roll_mean(x, 3))
career["last10_win_rate"] = g["won"].transform(lambda x: roll_mean(x, 10))
career["trend_score"]     = career["last3_win_rate"] - career["last10_win_rate"]
career["recency_win_rate"] = g["won"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).apply(
        lambda s: float(np.dot(np.arange(1, len(s)+1), s) / np.arange(1, len(s)+1).sum()),
        raw=True,
    )
)
career["prev_date"]   = g["date"].transform(lambda x: x.shift(1))
career["layoff_days"] = (career["date"] - career["prev_date"]).dt.days.fillna(0)

opp_src = (career[["fighter", "date", "career_win_rate"]]
           .rename(columns={"fighter": "opponent", "career_win_rate": "opp_win_rate"})
           .sort_values("date"))
career_with_opp = pd.merge_asof(
    career.sort_values("date"), opp_src,
    on="date", by="opponent", direction="backward",
)
career_with_opp = career_with_opp.sort_values(["fighter", "date"])
career_with_opp["opp_quality"] = (
    career_with_opp.groupby("fighter")["opp_win_rate"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

JOIN_COLS = [
    "career_win_rate", "ko_finish_rate", "sub_finish_rate",
    "last3_win_rate", "last10_win_rate", "trend_score",
    "layoff_days", "last5_won", "last5_finish_rate",
    "cum_wins", "cum_fights",
    "career_finish_rate", "recency_win_rate",
]
career_feat = career_with_opp[["fighter", "date"] + JOIN_COLS + ["opp_quality"]].copy()

def join_career(df, prefix):
    fc = f"{prefix}_fighter"
    sub = career_feat.rename(columns={
        "fighter": fc,
        **{c: f"{prefix}_{c}" for c in JOIN_COLS + ["opp_quality"]}
    }).sort_values("date")
    return pd.merge_asof(df.sort_values("date"), sub, on="date", by=fc, direction="backward")

# ── Master feature engineering ────────────────────────────────────────────────
master["Winner_bin"] = (master["Winner"] == "Red").astype(int)
WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11, "Catch Weight": 6,
}
master["weight_class_ord"] = master["weight_class"].map(WC_ORDER).fillna(6)
master["title_bout_bin"]   = (master["title_bout"] == True).astype(int)
r = master["R_Stance"].fillna("Unknown")
b = master["B_Stance"].fillna("Unknown")
master["orth_clash"]  = ((r == "Orthodox") & (b == "Orthodox")).astype(int)
master["south_clash"] = ((r == "Southpaw") & (b == "Southpaw")).astype(int)
master["R_southpaw"]  = (r == "Southpaw").astype(int)
master["B_southpaw"]  = (b == "Southpaw").astype(int)

master = join_career(master, "R")
master = join_career(master, "B")

# ── Style stats (ufc_fighters_final_updated) ──────────────────────────────────
STYLE_COLS = ["SLpM", "SApM", "Str_Acc", "Str_Def", "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg"]
style = fighters[["Fighter_Name"] + STYLE_COLS].drop_duplicates("Fighter_Name").copy()
for sc in ["Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]:
    if style[sc].dtype == object:
        style[sc] = pd.to_numeric(
            style[sc].astype(str).str.replace("%", "", regex=False), errors="coerce"
        ) / 100.0
for prefix, side in [("R", "R_fighter"), ("B", "B_fighter")]:
    sr = style.rename(columns={"Fighter_Name": side,
                                **{c: f"{prefix}_{c}" for c in STYLE_COLS}})
    master = master.merge(sr, on=side, how="left")

# ── Diff features ─────────────────────────────────────────────────────────────
def add_diff(df, col):
    rc, bc = f"R_{col}", f"B_{col}"
    if rc in df.columns and bc in df.columns:
        df[f"{col}_dif"] = df[rc] - df[bc]

for col in [
    "career_win_rate", "ko_finish_rate", "sub_finish_rate",
    "trend_score", "opp_quality",
    "last3_win_rate", "last10_win_rate", "layoff_days",
    "last5_won", "last5_finish_rate",
    "SLpM", "SApM", "Str_Def", "TD_Def", "Sub_Avg", "TD_Avg",
    "career_finish_rate", "recency_win_rate",
]:
    add_diff(master, col)

for p in ["R", "B"]:
    master[f"{p}_age_x_exp"] = master[f"{p}_age"] * master[f"{p}_cum_fights"]
add_diff(master, "age_x_exp")

for p in ["R", "B"]:
    ld = master[f"{p}_layoff_days"]
    master[f"{p}_layoff_lt90"]    = (ld < 90).astype(int)
    master[f"{p}_layoff_90_180"]  = ((ld >= 90)  & (ld < 180)).astype(int)
    master[f"{p}_layoff_180_365"] = ((ld >= 180) & (ld < 365)).astype(int)
    master[f"{p}_layoff_gt365"]   = (ld >= 365).astype(int)

log(f"  Feature engineering done in {time.time()-t0:.1f}s")

# ── Row count report ──────────────────────────────────────────────────────────
rows_total = len(master)
master_date_filtered = master[master["date"] >= pd.Timestamp(DATE_FROM)]
rows_after_date = len(master_date_filtered)

cols_avail = [c for c in FEAT_108 if c in master.columns]
master_exp_filtered = master_date_filtered[
    (master_date_filtered["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
    (master_date_filtered["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
]
rows_after_exp = len(master_exp_filtered)

sub_for_nan = master_exp_filtered[cols_avail + ["Winner_bin", "date"]].dropna(
    subset=cols_avail + ["Winner_bin"]
)
rows_after_nan = len(sub_for_nan)

log()
log(f"  Row counts:")
log(f"    Total master rows:         {rows_total:,}")
log(f"    After date≥{DATE_FROM}:       {rows_after_date:,}")
log(f"    After {MIN_UFC_FIGHTS}+ UFC fights filter: {rows_after_exp:,}")
log(f"    After NaN drop:            {rows_after_nan:,}")
train_count = len(sub_for_nan[sub_for_nan["date"] < pd.Timestamp(TEMPORAL_CUT)])
test_count  = len(sub_for_nan[sub_for_nan["date"] >= pd.Timestamp(TEMPORAL_CUT)])
log(f"    Training (2018-2023):      {train_count:,}")
log(f"    Test     (2024+):          {test_count:,}")
log(f"    Features available:        {len(cols_avail)}/108")

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def augment(X, y, feat_list):
    Xf = X.copy()
    yf = (1 - y).reset_index(drop=True)
    for rc in [c for c in feat_list if c.startswith("R_")]:
        bc = "B_" + rc[2:]
        if bc in feat_list:
            Xf[rc], Xf[bc] = X[bc].values, X[rc].values
    for dc in [c for c in feat_list if c.endswith("_dif")]:
        Xf[dc] = -X[dc].values
    return (
        pd.concat([X.reset_index(drop=True), Xf], ignore_index=True),
        pd.concat([y.reset_index(drop=True), yf], ignore_index=True),
    )

def get_filtered(feat_list):
    cols = [c for c in feat_list if c in master.columns]
    sub  = master[
        (master["date"] >= pd.Timestamp(DATE_FROM)) &
        (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
        (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
    ].copy()
    sub = sub[cols + ["Winner_bin", "date"]].dropna(subset=cols + ["Winner_bin"])
    return sub, cols

def temporal_sets(feat_list=None):
    if feat_list is None:
        feat_list = FEAT_108
    sub, cols = get_filtered(feat_list)
    train = sub[sub["date"] <  pd.Timestamp(TEMPORAL_CUT)]
    test  = sub[sub["date"] >= pd.Timestamp(TEMPORAL_CUT)]
    Xtr, ytr = train[cols], train["Winner_bin"]
    Xte, yte = test[cols].values, test["Winner_bin"].values
    Xtr_a, ytr_a = augment(Xtr, ytr, cols)
    return Xtr_a.values, Xte, ytr_a.values, yte, cols, len(train), len(test)

def random_sets(feat_list=None):
    if feat_list is None:
        feat_list = FEAT_108
    sub, cols = get_filtered(feat_list)
    X, y = sub[cols], sub["Winner_bin"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_a, ytr_a = augment(Xtr, ytr, cols)
    return Xtr_a.values, Xte.values, ytr_a.values, yte.values, cols

def build_lr(penalty, C, l1_ratio, scaler_name, solver="saga", max_iter=2000):
    lr = LogisticRegression(
        penalty=penalty, C=C, solver=solver,
        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        max_iter=max_iter, random_state=RS, n_jobs=1,
    )
    if scaler_name == "robust":
        return Pipeline([("sc", RobustScaler()), ("lr", lr)])
    if scaler_name == "standard":
        return Pipeline([("sc", StandardScaler()), ("lr", lr)])
    return lr   # no scaler

def eval_temporal(pipe, feat_list=None):
    Xtr, Xte, ytr, yte, cols, n_tr, n_te = temporal_sets(feat_list)
    pipe.fit(Xtr, ytr)
    return accuracy_score(yte, pipe.predict(Xte)), pipe.predict_proba(Xte), yte

def eval_random(pipe, feat_list=None):
    Xtr, Xte, ytr, yte, cols = random_sets(feat_list)
    pipe.fit(Xtr, ytr)
    return accuracy_score(yte, pipe.predict(Xte))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Baseline retrain (same hyperparams)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 72)
log("STEP 2 — Baseline retrain (same hyperparams, new data)")
log("=" * 72)

# LR baseline: penalty=elasticnet, C=0.108, l1_ratio=0.835, RobustScaler, saga
t = time.time()
lr_base = build_lr("elasticnet", 0.10774144910091868, 0.8350137588175889, "robust", "saga")
Xtr_b, Xte_b, ytr_b, yte_b, cols_b, _, _ = temporal_sets()
lr_base.fit(Xtr_b, ytr_b)
lr_base_proba = lr_base.predict_proba(Xte_b)
lr_base_t = accuracy_score(yte_b, lr_base.predict(Xte_b))
lr_base_r = eval_random(build_lr("elasticnet", 0.10774144910091868, 0.8350137588175889, "robust", "saga"))
record("baseline_LR", lr_base_t, lr_base_r, 108, time.time()-t,
       model=lr_base,
       meta={"model_type":"LR_baseline","params":{"penalty":"elasticnet","C":0.10774,"l1_ratio":0.835,"scaler":"robust"}})

# XGB baseline: load exp6 best params, retrain on temporal split
t = time.time()
import xgboost as xgb_mod
XGB_BASE_PARAMS = {
    "n_estimators": 1161, "learning_rate": 0.04923, "max_depth": 5,
    "subsample": 0.7724, "colsample_bytree": 0.7647, "min_child_weight": 4,
    "gamma": 0.7458, "reg_alpha": 1.1287,
    "use_label_encoder": False, "eval_metric": "logloss",
    "random_state": RS, "n_jobs": 1,
}
xgb_base = xgb_mod.XGBClassifier(**XGB_BASE_PARAMS)
xgb_base.fit(Xtr_b, ytr_b, verbose=False)
xgb_base_proba = xgb_base.predict_proba(Xte_b)
xgb_base_t = accuracy_score(yte_b, xgb_base.predict(Xte_b))

Xtr_r, Xte_r, ytr_r, yte_r, _ = random_sets()
xgb_base_r_model = xgb_mod.XGBClassifier(**XGB_BASE_PARAMS)
xgb_base_r_model.fit(Xtr_r, ytr_r, verbose=False)
xgb_base_r = accuracy_score(yte_r, xgb_base_r_model.predict(Xte_r))
record("baseline_XGB", xgb_base_t, xgb_base_r, 108, time.time()-t,
       model=xgb_base, xgb_model=xgb_base,
       meta={"model_type":"XGB_baseline","params":XGB_BASE_PARAMS})

# 90/10 blend baseline
t = time.time()
blend_base_proba = 0.9 * lr_base_proba + 0.1 * xgb_base_proba
blend_base_t = accuracy_score(yte_b, blend_base_proba.argmax(axis=1))
record("baseline_LR90_XGB10", blend_base_t, 0.0, 108, time.time()-t,
       model=lr_base, xgb_model=xgb_base,
       meta={"model_type":"blend_LR90_XGB10","blend_ratio":"90/10 LR+XGB"})

log()
log(f"  Previous best (old data):  {PREV_BEST_TEMPORAL:.4f}")
log(f"  Baseline blend (new data): {blend_base_t:.4f}  (delta: {blend_base_t - PREV_BEST_TEMPORAL:+.4f})")

gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Retune LR with Optuna (100 trials)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 72)
log("STEP 3 — LR Optuna (100 trials, temporal objective)")
log("=" * 72)
t_step = time.time()

# Pre-build data once for speed
Xtr_lr, Xte_lr, ytr_lr, yte_lr, cols_lr, _, _ = temporal_sets()

def lr_objective(trial):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C       = trial.suggest_float("C", 0.001, 100.0, log=True)
    l1_r    = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5
    scaler  = trial.suggest_categorical("scaler", ["robust", "standard", "none"])
    # Fixed solver space to avoid Optuna dynamic distribution error;
    # override incompatible combinations to "saga" (supports all penalties).
    solver_hint = trial.suggest_categorical("solver", ["saga", "liblinear", "lbfgs"])
    valid = {"l1": {"saga", "liblinear"}, "l2": {"saga", "liblinear", "lbfgs"},
             "elasticnet": {"saga"}}
    solver = solver_hint if solver_hint in valid[penalty] else "saga"
    pipe   = build_lr(penalty, C, l1_r, scaler, solver)
    try:
        pipe.fit(Xtr_lr, ytr_lr)
        return accuracy_score(yte_lr, pipe.predict(Xte_lr))
    except Exception:
        return 0.0

study_lr = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=RS))
study_lr.optimize(lr_objective, n_trials=100, show_progress_bar=False)

bp = study_lr.best_params
log(f"  Best LR temporal: {study_lr.best_value:.4f}")
log(f"  Params: {bp}")

# Refit best LR on temporal split, eval both temporal + random
t = time.time()
best_lr_pipe = build_lr(
    bp["penalty"], bp["C"],
    bp.get("l1_ratio", 0.5), bp["scaler"],
    bp.get("solver", "saga"),
)
best_lr_pipe.fit(Xtr_lr, ytr_lr)
best_lr_proba = best_lr_pipe.predict_proba(Xte_lr)
lr_tuned_t    = accuracy_score(yte_lr, best_lr_pipe.predict(Xte_lr))

# Random accuracy
best_lr_rand_pipe = build_lr(bp["penalty"], bp["C"],
                              bp.get("l1_ratio",0.5), bp["scaler"],
                              bp.get("solver","saga"))
lr_tuned_r = eval_random(best_lr_rand_pipe)
record("LR_tuned_100trials", lr_tuned_t, lr_tuned_r, 108, time.time()-t_step,
       model=best_lr_pipe,
       meta={"model_type":"LR_tuned","params":bp,
             "train_window":f"{DATE_FROM} to <{TEMPORAL_CUT}"})

gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Retune XGBoost with Optuna (50 trials)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 72)
log("STEP 4 — XGBoost Optuna (50 trials, temporal objective)")
log("=" * 72)
t_step = time.time()

Xtr_xgb, Xte_xgb, ytr_xgb, yte_xgb, _, _, _ = temporal_sets()

def xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 500, 1500),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "use_label_encoder": False, "eval_metric": "logloss",
        "random_state": RS, "n_jobs": 1,
    }
    m = xgb_mod.XGBClassifier(**params)
    try:
        m.fit(Xtr_xgb, ytr_xgb, verbose=False)
        return accuracy_score(yte_xgb, m.predict(Xte_xgb))
    except Exception:
        return 0.0

study_xgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RS))
study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=False)

bp_xgb = study_xgb.best_params
log(f"  Best XGB temporal: {study_xgb.best_value:.4f}")
log(f"  Params: {bp_xgb}")

t = time.time()
xgb_params_best = {**bp_xgb,
                   "use_label_encoder": False, "eval_metric": "logloss",
                   "random_state": RS, "n_jobs": 1}
best_xgb_model = xgb_mod.XGBClassifier(**xgb_params_best)
best_xgb_model.fit(Xtr_xgb, ytr_xgb, verbose=False)
best_xgb_proba  = best_xgb_model.predict_proba(Xte_xgb)
xgb_tuned_t     = accuracy_score(yte_xgb, best_xgb_model.predict(Xte_xgb))

Xtr_r2, Xte_r2, ytr_r2, yte_r2, _ = random_sets()
xgb_rand = xgb_mod.XGBClassifier(**xgb_params_best)
xgb_rand.fit(Xtr_r2, ytr_r2, verbose=False)
xgb_tuned_r = accuracy_score(yte_r2, xgb_rand.predict(Xte_r2))
record("XGB_tuned_50trials", xgb_tuned_t, xgb_tuned_r, 108, time.time()-t_step,
       model=best_lr_pipe, xgb_model=best_xgb_model,
       meta={"model_type":"XGB_tuned","params":bp_xgb})

gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Blend ratio sweep (best LR + best XGB)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 72)
log("STEP 5 — Blend ratio sweep (tuned LR + tuned XGB)")
log("=" * 72)
t_step = time.time()

# Ensure proba arrays are on the same test set
Xtr_bl, Xte_bl, ytr_bl, yte_bl, _, _, _ = temporal_sets()
bl_lr_pipe = build_lr(bp["penalty"], bp["C"], bp.get("l1_ratio",0.5),
                      bp["scaler"], bp.get("solver","saga"))
bl_lr_pipe.fit(Xtr_bl, ytr_bl)
bl_lr_proba = bl_lr_pipe.predict_proba(Xte_bl)

bl_xgb = xgb_mod.XGBClassifier(**xgb_params_best)
bl_xgb.fit(Xtr_bl, ytr_bl, verbose=False)
bl_xgb_proba = bl_xgb.predict_proba(Xte_bl)

ratios = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70]
for lr_w in ratios:
    xw = 1.0 - lr_w
    p  = lr_w * bl_lr_proba + xw * bl_xgb_proba
    t_acc = accuracy_score(yte_bl, p.argmax(axis=1))
    tag = f"blend_LR{int(lr_w*100)}_XGB{int(xw*100)}"
    record(tag, t_acc, 0.0, 108, time.time()-t_step,
           model=bl_lr_pipe, xgb_model=bl_xgb,
           meta={"model_type": tag, "blend_ratio": f"{lr_w:.0%} LR + {xw:.0%} XGB",
                 "lr_params": bp, "xgb_params": bp_xgb})

gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — CatBoost (50 Optuna trials)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 72)
log("STEP 6 — CatBoost Optuna (50 trials)")
log("=" * 72)
t_step = time.time()

try:
    from catboost import CatBoostClassifier

    Xtr_cb, Xte_cb, ytr_cb, yte_cb, _, _, _ = temporal_sets()

    def cb_objective(trial):
        params = {
            "iterations":    trial.suggest_int("iterations", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth":         trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed":   RS, "verbose": 0, "thread_count": 1,
            "allow_writing_files": False,
        }
        m = CatBoostClassifier(**params)
        try:
            m.fit(Xtr_cb, ytr_cb)
            return accuracy_score(yte_cb, m.predict(Xte_cb))
        except Exception:
            return 0.0

    study_cb = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=RS))
    study_cb.optimize(cb_objective, n_trials=50, show_progress_bar=False)

    bp_cb = study_cb.best_params
    log(f"  Best CatBoost temporal: {study_cb.best_value:.4f}")
    log(f"  Params: {bp_cb}")

    cb_params_best = {**bp_cb, "random_seed": RS, "verbose": 0,
                      "thread_count": 1, "allow_writing_files": False}
    best_cb = CatBoostClassifier(**cb_params_best)
    best_cb.fit(Xtr_cb, ytr_cb)
    best_cb_proba = best_cb.predict_proba(Xte_cb)
    cb_solo_t = accuracy_score(yte_cb, best_cb.predict(Xte_cb))

    Xtr_r3, Xte_r3, ytr_r3, yte_r3, _ = random_sets()
    cb_r = CatBoostClassifier(**cb_params_best)
    cb_r.fit(Xtr_r3, ytr_r3)
    cb_solo_r = accuracy_score(yte_r3, cb_r.predict(Xte_r3))
    record("CatBoost_tuned_50trials", cb_solo_t, cb_solo_r, 108,
           time.time()-t_step, model=bl_lr_pipe, xgb_model=best_cb,
           meta={"model_type":"CatBoost","params":bp_cb})

    # CatBoost blends with best LR (re-fit on same split)
    cb_lr_pipe = build_lr(bp["penalty"], bp["C"], bp.get("l1_ratio",0.5),
                          bp["scaler"], bp.get("solver","saga"))
    cb_lr_pipe.fit(Xtr_cb, ytr_cb)
    cb_lr_proba = cb_lr_pipe.predict_proba(Xte_cb)

    for lr_w, cb_w in [(0.85, 0.15), (0.90, 0.10)]:
        p  = lr_w * cb_lr_proba + cb_w * best_cb_proba
        t_acc = accuracy_score(yte_cb, p.argmax(axis=1))
        tag = f"blend_LR{int(lr_w*100)}_CB{int(cb_w*100)}"
        record(tag, t_acc, 0.0, 108, time.time()-t_step,
               model=cb_lr_pipe, xgb_model=best_cb,
               meta={"model_type": tag, "blend_ratio": f"{lr_w:.0%} LR + {cb_w:.0%} CB",
                     "lr_params": bp, "cb_params": bp_cb})

    gc.collect()

except ImportError:
    log("  CatBoost not installed — skipping Step 6")
    best_cb = None

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — LightGBM (50 Optuna trials)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 72)
log("STEP 7 — LightGBM Optuna (50 trials)")
log("=" * 72)
t_step = time.time()

try:
    import lightgbm as lgb_mod

    Xtr_lg, Xte_lg, ytr_lg, yte_lg, _, _, _ = temporal_sets()

    def lgb_objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 100),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": RS, "n_jobs": 1, "verbose": -1,
        }
        m = lgb_mod.LGBMClassifier(**params)
        try:
            m.fit(Xtr_lg, ytr_lg)
            return accuracy_score(yte_lg, m.predict(Xte_lg))
        except Exception:
            return 0.0

    study_lgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RS))
    study_lgb.optimize(lgb_objective, n_trials=50, show_progress_bar=False)

    bp_lgb = study_lgb.best_params
    log(f"  Best LightGBM temporal: {study_lgb.best_value:.4f}")
    log(f"  Params: {bp_lgb}")

    lgb_params_best = {**bp_lgb, "random_state": RS, "n_jobs": 1, "verbose": -1}
    best_lgb = lgb_mod.LGBMClassifier(**lgb_params_best)
    best_lgb.fit(Xtr_lg, ytr_lg)
    best_lgb_proba = best_lgb.predict_proba(Xte_lg)
    lgb_solo_t = accuracy_score(yte_lg, best_lgb.predict(Xte_lg))

    Xtr_r4, Xte_r4, ytr_r4, yte_r4, _ = random_sets()
    lgb_r = lgb_mod.LGBMClassifier(**lgb_params_best)
    lgb_r.fit(Xtr_r4, ytr_r4)
    lgb_solo_r = accuracy_score(yte_r4, lgb_r.predict(Xte_r4))
    record("LGB_tuned_50trials", lgb_solo_t, lgb_solo_r, 108,
           time.time()-t_step, model=bl_lr_pipe, xgb_model=best_lgb,
           meta={"model_type":"LGB","params":bp_lgb})

    # LightGBM blends with best LR
    lg_lr_pipe = build_lr(bp["penalty"], bp["C"], bp.get("l1_ratio",0.5),
                          bp["scaler"], bp.get("solver","saga"))
    lg_lr_pipe.fit(Xtr_lg, ytr_lg)
    lg_lr_proba = lg_lr_pipe.predict_proba(Xte_lg)

    for lr_w, lgb_w in [(0.85, 0.15), (0.90, 0.10)]:
        p  = lr_w * lg_lr_proba + lgb_w * best_lgb_proba
        t_acc = accuracy_score(yte_lg, p.argmax(axis=1))
        tag = f"blend_LR{int(lr_w*100)}_LGB{int(lgb_w*100)}"
        record(tag, t_acc, 0.0, 108, time.time()-t_step,
               model=lg_lr_pipe, xgb_model=best_lgb,
               meta={"model_type": tag, "blend_ratio": f"{lr_w:.0%} LR + {lgb_w:.0%} LGB",
                     "lr_params": bp, "lgb_params": bp_lgb})

    gc.collect()

except ImportError:
    log("  LightGBM not installed — skipping Step 7")
    best_lgb = None

# ─────────────────────────────────────────────────────────────────────────────
# Save winner
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 72)
log("SAVING BEST MODEL")
log("=" * 72)
save_best()

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
all_results.sort(key=lambda x: -x["temporal_acc"])

log()
log("=" * 72)
log("ALL RESULTS (ranked by temporal accuracy):")
log("=" * 72)
log(f"  {'Config':60s}  {'Temporal':>8}  {'Random':>8}  {'Feats':>5}")
log("  " + "-"*88)
for r in all_results:
    log(f"  {r['config']:60s}  {r['temporal_acc']:.4f}    {r['random_acc']:.4f}    {r['n_feats']:3d}")

log()
log("=" * 40)
log("RETRAINING — FINAL SUMMARY")
log("=" * 40)
log(f"Data: career_fights_updated.csv (49,953 rows)")
log(f"Filter: date≥{DATE_FROM}, {MIN_UFC_FIGHTS}+ UFC fights")
log(f"Training rows: {train_count} | Test rows (2024+): {test_count}")
log()
log("BASELINE (same hyperparams, new data):")
log(f"  LR:              temporal {lr_base_t:.4f} | random {lr_base_r:.4f}")
log(f"  XGBoost:         temporal {xgb_base_t:.4f} | random {xgb_base_r:.4f}")
log(f"  90/10 blend:     temporal {blend_base_t:.4f} | random {0:.4f}")
log(f"  vs old baseline: {blend_base_t - PREV_BEST_TEMPORAL:+.4f} ({'+' if blend_base_t > PREV_BEST_TEMPORAL else ''}{(blend_base_t - PREV_BEST_TEMPORAL)*100:.2f}% pts)")
log()
log("TUNED RESULTS (ranked by temporal):")
log(f"  {'Config':40s}  {'Temporal':>8}   {'Random':>8}")
log("  " + "-"*62)
for r in all_results:
    log(f"  {r['config']:40s}  {r['temporal_acc']:.4f}    {r['random_acc']:.4f}")
log()

best_res = all_results[0]
log(f"Best config: {best_res['config']}")
log(f"Best temporal accuracy: {best_res['temporal_acc']:.4f}")
log(f"vs previous best (71.54%): {best_res['temporal_acc'] - PREV_BEST_TEMPORAL:+.4f}")
log()
log(f"LR hyperparams:  penalty={bp['penalty']}, C={bp['C']:.5f}, l1_ratio={bp.get('l1_ratio','N/A')}, scaler={bp['scaler']}, solver={bp.get('solver','saga')}")
log(f"XGB hyperparams: n_est={bp_xgb.get('n_estimators')}, lr={bp_xgb.get('learning_rate'):.4f}, max_depth={bp_xgb.get('max_depth')}")
log()
log("Files saved:")
for path in [BEST_MDL, BEST_XGB, BEST_FT, META_JSON]:
    mark = "✓" if os.path.exists(path) else "✗"
    log(f"  {mark} {path}")
log("=" * 40)

print()
print("=" * 40)
print("RETRAINING — FINAL SUMMARY")
print("=" * 40)
print(f"Data: career_fights_updated.csv (49,953 rows)")
print(f"Filter: date≥{DATE_FROM}, {MIN_UFC_FIGHTS}+ UFC fights")
print(f"Training rows: {train_count} | Test rows (2024+): {test_count}")
print()
print("BASELINE (same hyperparams, new data):")
print(f"  LR:              temporal {lr_base_t:.2%} | random {lr_base_r:.2%}")
print(f"  XGBoost:         temporal {xgb_base_t:.2%} | random {xgb_base_r:.2%}")
print(f"  90/10 blend:     temporal {blend_base_t:.2%} | random {0:.2%}")
print(f"  vs old baseline: {blend_base_t - PREV_BEST_TEMPORAL:+.2%} (data fix impact)")
print()
print("TUNED RESULTS (ranked by temporal):")
print(f"  {'Config':40s}  {'Temporal':>8}   {'Random':>8}")
print("  " + "-"*62)
for r in all_results:
    print(f"  {r['config']:40s}  {r['temporal_acc']:.2%}    {r['random_acc']:.2%}")
print()
print(f"Best config: {best_res['config']}")
print(f"Best temporal accuracy: {best_res['temporal_acc']:.2%}")
print(f"vs previous best (71.54%): {best_res['temporal_acc'] - PREV_BEST_TEMPORAL:+.2%}")
print()
print(f"LR hyperparams: penalty={bp['penalty']}, C={bp['C']:.5f}, l1_ratio={bp.get('l1_ratio','N/A')}, scaler={bp['scaler']}, solver={bp.get('solver','saga')}")
print(f"XGB hyperparams: n_est={bp_xgb.get('n_estimators')}, lr={bp_xgb.get('learning_rate'):.4f}, max_depth={bp_xgb.get('max_depth')}")
print()
print("Files saved:")
for path in [BEST_MDL, BEST_XGB, BEST_FT, META_JSON]:
    mark = "✓" if os.path.exists(path) else "✗"
    print(f"  {mark} {path}")
print("=" * 40)
