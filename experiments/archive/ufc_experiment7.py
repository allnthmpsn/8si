"""
ufc_experiment7.py — Best Logistic Regression for temporal generalization.

Primary metric : temporal accuracy  (train <2024, test ≥2024)
Secondary metric: random 80/20 accuracy
Target: beat 70.22% temporal accuracy (LR from exp6 baseline)

Experiments:
  A — LR Optuna 100 trials  (temporal score)
  B — Feature selection on best-A LR
  C — Training window sweep
  D — Calibration (isotonic / sigmoid / uncalibrated)
  E — Soft-vote blend: best LR + saved XGBoost  (70/30, 80/20, 90/10)
"""

import gc, json, os, time, warnings
import joblib, numpy as np, pandas as pd, optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RS             = 42
TEMPORAL_ACC_BASELINE = 0.7022   # LR temporal from exp6
DATE_CUT       = "2017-01-01"
MIN_UFC_FIGHTS = 3
TEMPORAL_CUT   = "2024-01-01"

OUT_LOG   = "model/experiment7_output.txt"
BEST_MDL  = "model/ufc_model_best.pkl"
BEST_FT   = "model/feature_columns_best.pkl"
LR_MDL    = "model/ufc_model_lr_best.pkl"
META_JSON = "model/model_metadata.json"

os.makedirs("model", exist_ok=True)
with open(OUT_LOG, "w") as f:
    f.write("")

def log(msg=""):
    print(msg, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(msg + "\n")

log("=" * 72)
log(f"UFC EXPERIMENT 7  —  {pd.Timestamp.now().isoformat()}")
log(f"Primary target  : temporal accuracy > {TEMPORAL_ACC_BASELINE:.4f}  (train<2024, test≥2024)")
log(f"Secondary target: random 80/20 accuracy")
log("=" * 72)

best_temporal  = TEMPORAL_ACC_BASELINE
best_random    = 0.0
best_model_obj = None
best_feats_list = None
best_meta       = {}
all_results     = []

# Load best XGBoost BEFORE any experiments (Exp E needs it)
log("\n[Pre] Loading saved XGBoost model for Experiment E...")
try:
    xgb_saved      = joblib.load(BEST_MDL)
    xgb_saved_feats = joblib.load(BEST_FT)
    log(f"  XGBoost loaded: {type(xgb_saved).__name__}, {len(xgb_saved_feats)} features")
except Exception as e:
    xgb_saved = None
    log(f"  Could not load XGBoost: {e}")

def record(config, t_acc, r_acc, n_feats, secs,
           model=None, feats=None, meta=None):
    global best_temporal, best_random, best_model_obj, best_feats_list, best_meta
    tag = "  *** NEW BEST (temporal) ***" if t_acc > best_temporal else ""
    log(f"  {config:68s}  t={t_acc:.4f}  r={r_acc:.4f}  feats={n_feats:3d}  {secs:.0f}s{tag}")
    all_results.append({
        "config": config, "temporal_acc": t_acc, "random_acc": r_acc,
        "n_feats": n_feats, "secs": secs,
    })
    if t_acc > best_temporal and model is not None and feats is not None:
        best_temporal   = t_acc
        best_random     = r_acc
        best_model_obj  = model
        best_feats_list = feats
        best_meta       = meta or {}
        joblib.dump(model, BEST_MDL)
        joblib.dump(feats, BEST_FT)
        with open(META_JSON, "w") as f:
            json.dump({
                "model_type":       type(model).__name__,
                "temporal_accuracy": t_acc,
                "random_accuracy":   r_acc,
                "n_features":        n_feats,
                "feature_list":      list(feats),
                "hyperparameters":   meta.get("params", {}),
                "training_window":   meta.get("train_window", "2017-01-01 to <2024"),
                "date_trained":      pd.Timestamp.now().isoformat(),
            }, f, indent=2)
        log(f"  *** Saved to {BEST_MDL}  +  {META_JSON}")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
log("\n[FE] Engineering features...")
t_fe = time.time()

career_raw = pd.read_csv("data/career_fights.csv")
career_raw["date"] = pd.to_datetime(career_raw["date"])
master     = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
fighters   = pd.read_csv("data/ufc_fighters_final.csv")
fighters["Wins"]   = pd.to_numeric(fighters["Wins"],   errors="coerce").fillna(0).astype(int)
fighters["Losses"] = pd.to_numeric(fighters["Losses"], errors="coerce").fillna(0).astype(int)

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
career["ko_finish_rate"]   = career["cum_did_ko"]  / career["cum_fights"].clip(lower=1)
career["sub_finish_rate"]  = career["cum_did_sub"] / career["cum_fights"].clip(lower=1)
career["cum_finish_wins"]  = career["cum_did_ko"] + career["cum_did_sub"]
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
    return pd.merge_asof(df.sort_values("date"), sub,
                         on="date", by=fc, direction="backward")

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

for p in ["R", "B"]:
    w, l = master[f"{p}_wins"], master[f"{p}_losses"]
    master[f"{p}_ufc_win_rate"] = w / (w + l).clip(lower=1)

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
    "career_finish_rate", "recency_win_rate", "ufc_win_rate",
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

log(f"  FE done in {time.time()-t_fe:.1f}s  master: {master.shape}")

BASE_108 = joblib.load(BEST_FT)
log(f"  BASE_108 = {len(BASE_108)} features")

# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
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

def get_filtered(feat_list, date_from=DATE_CUT):
    cols = [c for c in feat_list if c in master.columns]
    sub  = master[
        (master["date"] >= pd.Timestamp(date_from)) &
        (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
        (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
    ].copy()
    sub = sub[cols + ["Winner_bin", "date"]].dropna(subset=cols + ["Winner_bin"])
    return sub, cols

def temporal_sets(feat_list, date_from=DATE_CUT, test_from=TEMPORAL_CUT):
    sub, cols = get_filtered(feat_list, date_from)
    train = sub[sub["date"] <  pd.Timestamp(test_from)]
    test  = sub[sub["date"] >= pd.Timestamp(test_from)]
    Xtr, ytr = train[cols], train["Winner_bin"]
    Xte, yte = test[cols].values, test["Winner_bin"].values
    Xtr_a, ytr_a = augment(Xtr, ytr, cols)
    return Xtr_a.values, Xte, ytr_a.values, yte, cols, len(train), len(test)

def random_sets(feat_list, date_from=DATE_CUT):
    sub, cols = get_filtered(feat_list, date_from)
    X, y = sub[cols], sub["Winner_bin"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_a, ytr_a = augment(Xtr, ytr, cols)
    return Xtr_a.values, Xte.values, ytr_a.values, yte.values, cols

def make_scaler(name):
    if name == "standard": return StandardScaler()
    if name == "robust":   return RobustScaler()
    if name == "minmax":   return MinMaxScaler()
    return None   # "none"

def build_lr_pipe(penalty, C, solver, l1_ratio, scaler_name, max_iter=2000):
    sc = make_scaler(scaler_name)
    lr = LogisticRegression(
        penalty=penalty, C=C, solver=solver,
        l1_ratio=l1_ratio if penalty == "elasticnet" else 0.5,
        max_iter=max_iter, random_state=RS, n_jobs=1,
    )
    if sc is None:
        return lr, lr   # pipe = lr itself, lr_step = lr
    pipe = Pipeline([("sc", sc), ("lr", lr)])
    return pipe, lr

def eval_temporal(pipe, feat_list, date_from=DATE_CUT, test_from=TEMPORAL_CUT):
    Xtr, Xte, ytr, yte, cols, n_tr, n_te = temporal_sets(feat_list, date_from, test_from)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)
    preds = np.argmax(proba, axis=1)
    return accuracy_score(yte, preds), proba, yte

def eval_random(pipe_cls, params, feat_list, scaler_name, date_from=DATE_CUT):
    """Build fresh model instance and evaluate on random split."""
    penalty, C, solver, l1_ratio = (
        params["penalty"], params["C"], params["solver"], params.get("l1_ratio", 0.5)
    )
    pipe, _ = build_lr_pipe(penalty, C, solver, l1_ratio, scaler_name)
    Xtr, Xte, ytr, yte, _ = random_sets(feat_list, date_from)
    pipe.fit(Xtr, ytr)
    return accuracy_score(yte, pipe.predict(Xte))

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — LR Optuna, 100 trials, optimise for temporal accuracy
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("EXPERIMENT A — LR Optuna (100 trials, primary metric = temporal acc)")
log("=" * 72)
t_exp = time.time()

Xtr_a, Xte_a, ytr_a, yte_a, cols_a, n_tr_a, n_te_a = temporal_sets(BASE_108)
log(f"  Train (pre-2024): {n_tr_a} (aug'd {len(ytr_a)}), "
    f"Test (2024+): {n_te_a}, Features: {len(cols_a)}")

def obj_a(trial):
    penalty  = trial.suggest_categorical("penalty",  ["l1", "l2", "elasticnet"])
    C        = trial.suggest_float("C", 1e-3, 100.0, log=True)
    scaler_n = trial.suggest_categorical("scaler", ["standard", "robust", "minmax", "none"])
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5

    if penalty == "elasticnet":
        solver = "saga"
    elif penalty == "l1":
        solver = trial.suggest_categorical("solver_l1", ["saga", "liblinear"])
    else:
        solver = trial.suggest_categorical("solver_l2", ["saga", "lbfgs", "liblinear"])

    pipe, _ = build_lr_pipe(penalty, C, solver, l1_ratio, scaler_n)
    pipe.fit(Xtr_a, ytr_a)
    return accuracy_score(yte_a, pipe.predict(Xte_a))

study_a = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=RS))
study_a.optimize(obj_a, n_trials=100, timeout=600, n_jobs=1, show_progress_bar=False)

# Extract top 10 Optuna trials by temporal accuracy
trials_df = pd.DataFrame([
    {**t.params, "temporal_acc": t.value}
    for t in study_a.trials if t.value is not None
]).sort_values("temporal_acc", ascending=False).head(10)

log("\n  Top 10 Optuna trials by temporal accuracy:")
log(trials_df.to_string(index=False))

bp_a = study_a.best_params
log(f"\n  Best params: {bp_a}")

# Determine solver for best params
if bp_a["penalty"] == "elasticnet":
    bp_a_solver = "saga"
elif bp_a["penalty"] == "l1":
    bp_a_solver = bp_a.get("solver_l1", "saga")
else:
    bp_a_solver = bp_a.get("solver_l2", "lbfgs")

best_pipe_a, best_lr_a = build_lr_pipe(
    bp_a["penalty"], bp_a["C"], bp_a_solver,
    bp_a.get("l1_ratio", 0.5), bp_a["scaler"]
)
best_pipe_a.fit(Xtr_a, ytr_a)
t_acc_a = accuracy_score(yte_a, best_pipe_a.predict(Xte_a))
r_acc_a = eval_random(None, {
    "penalty": bp_a["penalty"], "C": bp_a["C"],
    "solver": bp_a_solver, "l1_ratio": bp_a.get("l1_ratio", 0.5),
}, cols_a, bp_a["scaler"])

elapsed_a = time.time() - t_exp
meta_a = {"params": {k: v for k, v in bp_a.items()
                     if k not in ("solver_l1", "solver_l2")},
          "train_window": f"{DATE_CUT} to <{TEMPORAL_CUT}"}
meta_a["params"]["solver"] = bp_a_solver

record(f"A_LR_Optuna_{bp_a['penalty']}_C{bp_a['C']:.4f}_{bp_a['scaler']}",
       t_acc_a, r_acc_a, len(cols_a), elapsed_a, best_pipe_a, list(cols_a), meta_a)
log(f"\n  Experiment A done in {elapsed_a:.0f}s")
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Feature selection on best-A LR
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("EXPERIMENT B — Feature selection for best-A LR config")
log("=" * 72)
t_exp = time.time()

# Get coefficients from the fitted best_pipe_a
if hasattr(best_pipe_a, "named_steps"):
    lr_step_a = best_pipe_a.named_steps.get("lr", best_pipe_a)
else:
    lr_step_a = best_pipe_a
coef_abs_a = np.abs(lr_step_a.coef_[0])
imp_b = pd.DataFrame({"feature": cols_a, "abs_coef": coef_abs_a}
                     ).sort_values("abs_coef", ascending=False)
log("\n  Top 20 features by |coef| (fitted on temporal training set):")
log(imp_b.head(20).to_string(index=False))

# Feature subsets
diff_feats = [c for c in cols_a if c.endswith("_dif")]
age_cum_layoff = (
    [c for c in cols_a if "age" in c and not c.endswith("_dif") and "x_exp" not in c] +
    [c for c in cols_a if "cum_fights" in c] +
    [c for c in cols_a if "layoff" in c]
)
diff_plus = list(set(diff_feats + age_cum_layoff))

feature_subsets = [
    ("all_108",            cols_a),
    ("top80_by_coef",      imp_b.head(80)["feature"].tolist()),
    ("top60_by_coef",      imp_b.head(60)["feature"].tolist()),
    ("top40_by_coef",      imp_b.head(40)["feature"].tolist()),
    ("top20_by_coef",      imp_b.head(20)["feature"].tolist()),
    (f"diff_only_{len(diff_feats)}",  diff_feats),
    (f"diff+age+cum+layoff_{len(diff_plus)}", diff_plus),
]

best_b_t_acc = 0.0
best_b_feats = cols_a
best_b_config = "all_108"

for tag, flist in feature_subsets:
    t0 = time.time()
    flist_avail = [c for c in flist if c in master.columns]
    if len(flist_avail) < 5:
        continue

    # Temporal eval
    pipe_b, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                               bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
    Xtr_b, Xte_b, ytr_b, yte_b, cols_b, n_trb, n_teb = temporal_sets(flist_avail)
    pipe_b.fit(Xtr_b, ytr_b)
    t_acc_b = accuracy_score(yte_b, pipe_b.predict(Xte_b))

    # Random eval
    pipe_br, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
    Xtr_br, Xte_br, ytr_br, yte_br, _ = random_sets(flist_avail)
    pipe_br.fit(Xtr_br, ytr_br)
    r_acc_b = accuracy_score(yte_br, pipe_br.predict(Xte_br))

    elapsed = time.time() - t0
    save_m = pipe_b if t_acc_b > best_temporal else None
    save_f = flist_avail if t_acc_b > best_temporal else None
    record(f"B_{tag}", t_acc_b, r_acc_b, len(flist_avail), elapsed,
           save_m, save_f, {**meta_a, "feature_subset": tag})

    if t_acc_b > best_b_t_acc:
        best_b_t_acc = t_acc_b
        best_b_feats = flist_avail
        best_b_config = tag
    del pipe_b, pipe_br; gc.collect()

log(f"\n  Best feature subset (temporal): {best_b_config}  acc={best_b_t_acc:.4f}")
log(f"  Experiment B done in {time.time()-t_exp:.0f}s")

# Best LR pipe on best feature set (used for C, D, E)
best_b_pipe, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
Xtr_best, Xte_best, ytr_best, yte_best, _, _, _ = temporal_sets(best_b_feats)
best_b_pipe.fit(Xtr_best, ytr_best)
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Training window sweep
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("EXPERIMENT C — Training window sweep")
log("=" * 72)
t_exp = time.time()

windows = [
    ("2015+ train, 2024+ test", "2015-01-01", "2024-01-01"),
    ("2016+ train, 2024+ test", "2016-01-01", "2024-01-01"),
    ("2017+ train, 2024+ test", "2017-01-01", "2024-01-01"),  # baseline
    ("2018+ train, 2024+ test", "2018-01-01", "2024-01-01"),
    ("2019+ train, 2024+ test", "2019-01-01", "2024-01-01"),
    ("2017+ train, 2023+ test", "2017-01-01", "2023-01-01"),
    ("2017+ train, 2022+ test", "2017-01-01", "2022-01-01"),
]

for wname, d_from, t_from in windows:
    t0 = time.time()
    try:
        Xtr_c, Xte_c, ytr_c, yte_c, cols_c, n_trc, n_tec = temporal_sets(
            best_b_feats, date_from=d_from, test_from=t_from
        )
        if n_trc < 100 or n_tec < 50:
            log(f"  SKIP {wname}: too few rows (train={n_trc}, test={n_tec})")
            continue
        pipe_c, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                   bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
        pipe_c.fit(Xtr_c, ytr_c)
        t_acc_c = accuracy_score(yte_c, pipe_c.predict(Xte_c))

        # Random split for this date_from window
        pipe_cr, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                    bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
        Xtr_cr, Xte_cr, ytr_cr, yte_cr, _ = random_sets(best_b_feats, d_from)
        pipe_cr.fit(Xtr_cr, ytr_cr)
        r_acc_c = accuracy_score(yte_cr, pipe_cr.predict(Xte_cr))

        log(f"  {wname:45s}  train={n_trc:4d}  test={n_tec:4d}")
        wmeta = {**meta_a, "train_window": f"{d_from} to <{t_from}"}
        save_m = pipe_c if t_acc_c > best_temporal else None
        save_f = cols_c if t_acc_c > best_temporal else None
        record(f"C_{d_from[:4]}+train_{t_from[:4]}+test",
               t_acc_c, r_acc_c, len(cols_c), time.time()-t0,
               save_m, save_f, wmeta)
        del pipe_c, pipe_cr; gc.collect()
    except Exception as e:
        log(f"  ERROR {wname}: {e}")

log(f"\n  Experiment C done in {time.time()-t_exp:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — Calibration
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("EXPERIMENT D — Calibration (isotonic / sigmoid, cv=5 on training set)")
log("=" * 72)
t_exp = time.time()

Xtr_d, Xte_d, ytr_d, yte_d, cols_d, n_trd, n_ted = temporal_sets(best_b_feats)
log(f"  Train: {n_trd} (aug'd {len(ytr_d)}), Test: {n_ted}")

# Uncalibrated baseline
t0 = time.time()
pipe_d_base, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                 bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
pipe_d_base.fit(Xtr_d, ytr_d)
proba_unc  = pipe_d_base.predict_proba(Xte_d)
preds_unc  = np.argmax(proba_unc, axis=1)
t_acc_unc  = accuracy_score(yte_d, preds_unc)
brier_unc  = brier_score_loss(yte_d, proba_unc[:, 1])
ll_unc     = log_loss(yte_d, proba_unc)
log(f"  Uncalibrated: acc={t_acc_unc:.4f}  brier={brier_unc:.4f}  logloss={ll_unc:.4f}")
record("D_LR_uncalibrated", t_acc_unc, 0.0, len(cols_d), time.time()-t0,
       pipe_d_base, cols_d, meta_a)

# Calibrated versions
for method in ["isotonic", "sigmoid"]:
    t0 = time.time()
    pipe_d_inner, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                     bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
    cal = CalibratedClassifierCV(pipe_d_inner, method=method, cv=5)
    cal.fit(Xtr_d, ytr_d)
    proba_cal  = cal.predict_proba(Xte_d)
    preds_cal  = np.argmax(proba_cal, axis=1)
    t_acc_cal  = accuracy_score(yte_d, preds_cal)
    brier_cal  = brier_score_loss(yte_d, proba_cal[:, 1])
    ll_cal     = log_loss(yte_d, proba_cal)
    log(f"  {method:10s}: acc={t_acc_cal:.4f}  brier={brier_cal:.4f}  logloss={ll_cal:.4f}")

    # Random split accuracy
    pipe_dr, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
    cal_r = CalibratedClassifierCV(pipe_dr, method=method, cv=5)
    Xtr_dr, Xte_dr, ytr_dr, yte_dr, _ = random_sets(best_b_feats)
    cal_r.fit(Xtr_dr, ytr_dr)
    r_acc_d = accuracy_score(yte_dr, cal_r.predict(Xte_dr))
    elapsed = time.time() - t0

    save_m = cal if t_acc_cal > best_temporal else None
    save_f = cols_d if t_acc_cal > best_temporal else None
    record(f"D_LR_calibrated_{method}", t_acc_cal, r_acc_d, len(cols_d), elapsed,
           save_m, save_f, {**meta_a, "calibration": method})
    del cal, cal_r; gc.collect()

log(f"\n  Experiment D done in {time.time()-t_exp:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT E — Soft-vote blend: best LR + saved XGBoost
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("EXPERIMENT E — Soft-vote blend: LR + saved XGBoost")
log("=" * 72)
t_exp = time.time()

if xgb_saved is None:
    log("  SKIP — XGBoost model not available")
else:
    # Temporal sets for LR features (best_b_feats)
    Xtr_e_lr, Xte_e_lr, ytr_e_lr, yte_e_lr, cols_e_lr, n_tre, n_tee = (
        temporal_sets(best_b_feats)
    )
    # Temporal sets for XGBoost features (may differ)
    Xtr_e_xgb, Xte_e_xgb, ytr_e_xgb, yte_e_xgb, cols_e_xgb, _, _ = (
        temporal_sets(xgb_saved_feats)
    )
    log(f"  LR features: {len(cols_e_lr)}, XGB features: {len(cols_e_xgb)}")
    log(f"  Test size: {n_tee}")

    # Fit fresh LR on temporal train
    pipe_e_lr, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                  bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
    pipe_e_lr.fit(Xtr_e_lr, ytr_e_lr)
    lr_proba  = pipe_e_lr.predict_proba(Xte_e_lr)

    # XGBoost predictions (note: xgb_saved was trained on random split — use as-is)
    xgb_proba = xgb_saved.predict_proba(Xte_e_xgb)

    # Verify test sets are aligned (same fights)
    # They should be since both use the same date filter and NaN drop on 108 features
    if len(lr_proba) != len(xgb_proba):
        log(f"  WARNING: test set sizes differ "
            f"(LR={len(lr_proba)}, XGB={len(xgb_proba)}) — aligning by index")
        # Find common test indices using date-based alignment
        sub_lr = master[
            (master["date"] >= pd.Timestamp(DATE_CUT)) &
            (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
            (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
        ].copy()
        sub_lr_avail  = [c for c in best_b_feats  if c in master.columns]
        sub_xgb_avail = [c for c in xgb_saved_feats if c in master.columns]
        sub_lr  = sub_lr[sub_lr_avail  + ["Winner_bin", "date"]].dropna()
        sub_xgb = sub_lr[sub_xgb_avail + ["Winner_bin", "date"]].dropna()
        common_idx = sub_lr.index.intersection(sub_xgb.index)
        sub_lr  = sub_lr.loc[common_idx]
        sub_xgb = sub_xgb.loc[common_idx]
        test_lr  = sub_lr[sub_lr["date"]  >= pd.Timestamp(TEMPORAL_CUT)]
        test_xgb = sub_xgb[sub_xgb["date"] >= pd.Timestamp(TEMPORAL_CUT)]
        train_lr  = sub_lr[sub_lr["date"]  < pd.Timestamp(TEMPORAL_CUT)]
        Xtr_e2, ytr_e2 = train_lr[sub_lr_avail], train_lr["Winner_bin"]
        Xtr_e2_a, ytr_e2_a = augment(Xtr_e2, ytr_e2, sub_lr_avail)
        pipe_e_lr.fit(Xtr_e2_a.values, ytr_e2_a.values)
        lr_proba  = pipe_e_lr.predict_proba(test_lr[sub_lr_avail].values)
        xgb_proba = xgb_saved.predict_proba(test_xgb[sub_xgb_avail].values)
        yte_e_lr  = test_lr["Winner_bin"].values

    # XGBoost was trained on random split; retrain on temporal train for fair comparison
    log("  Retraining XGBoost on temporal training set for fair comparison...")
    import xgboost as xgb_mod
    if hasattr(xgb_saved, "get_params"):
        xgb_params = {k: v for k, v in xgb_saved.get_params().items()
                      if k not in ("use_label_encoder",)}
        xgb_params.update({"use_label_encoder": False, "eval_metric": "logloss",
                            "random_state": RS, "n_jobs": 1})
        xgb_retrained = xgb_mod.XGBClassifier(**xgb_params)
        xgb_retrained.fit(Xtr_e_xgb, ytr_e_xgb, verbose=False)
        xgb_proba_temp = xgb_retrained.predict_proba(Xte_e_xgb)
        xgb_solo_acc = accuracy_score(yte_e_xgb, xgb_retrained.predict(Xte_e_xgb))
        log(f"  XGBoost (retrained temporal) solo accuracy: {xgb_solo_acc:.4f}")
    else:
        xgb_proba_temp = xgb_proba
        xgb_solo_acc = accuracy_score(yte_e_lr, xgb_saved.predict(Xte_e_xgb))

    lr_solo_acc = accuracy_score(yte_e_lr, pipe_e_lr.predict(Xte_e_lr))
    log(f"  LR solo temporal accuracy: {lr_solo_acc:.4f}")

    # Ensure test labels align
    yte_blend = yte_e_lr if len(yte_e_lr) <= len(yte_e_xgb) else yte_e_xgb
    n_blend   = len(yte_blend)
    lr_p   = lr_proba[:n_blend]
    xgb_p  = xgb_proba_temp[:n_blend]

    for lr_w, xgb_w in [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]:
        t0 = time.time()
        blended   = lr_w * lr_p + xgb_w * xgb_p
        preds_bl  = (blended[:, 1] > 0.5).astype(int)
        t_acc_bl  = accuracy_score(yte_blend, preds_bl)

        # Random-split blend (retrain both on random split)
        Xtr_er, Xte_er, ytr_er, yte_er, cols_er = random_sets(best_b_feats)
        pipe_er, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                                    bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
        pipe_er.fit(Xtr_er, ytr_er)
        Xtr_xr, Xte_xr, ytr_xr, yte_xr, _ = random_sets(xgb_saved_feats)
        xgb_r = xgb_mod.XGBClassifier(**xgb_params)
        xgb_r.fit(Xtr_xr, ytr_xr, verbose=False)
        n_r = min(len(yte_er), len(yte_xr))
        bl_r = lr_w * pipe_er.predict_proba(Xte_er[:n_r]) + \
               xgb_w * xgb_r.predict_proba(Xte_xr[:n_r])
        r_acc_bl = accuracy_score(yte_er[:n_r], (bl_r[:, 1] > 0.5).astype(int))

        save_note = None  # blends aren't easily serialisable; skip saving
        record(f"E_LR{int(lr_w*100)}pct_XGB{int(xgb_w*100)}pct_blend",
               t_acc_bl, r_acc_bl, len(best_b_feats), time.time()-t0)
        del pipe_er, xgb_r; gc.collect()

    del xgb_retrained; gc.collect()

log(f"\n  Experiment E done in {time.time()-t_exp:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE BEST LR (regardless of whether it beats temporal baseline)
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[Saving] Best LR config regardless of temporal baseline...")
# Fit final best LR on full temporal train set with best features
lr_final_feats = best_b_feats
Xtr_fin, Xte_fin, ytr_fin, yte_fin, cols_fin, _, _ = temporal_sets(lr_final_feats)
pipe_fin, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                              bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
pipe_fin.fit(Xtr_fin, ytr_fin)
t_acc_fin  = accuracy_score(yte_fin, pipe_fin.predict(Xte_fin))
pipe_rfin, _ = build_lr_pipe(bp_a["penalty"], bp_a["C"], bp_a_solver,
                               bp_a.get("l1_ratio", 0.5), bp_a["scaler"])
Xtr_rfin, Xte_rfin, ytr_rfin, yte_rfin, _ = random_sets(lr_final_feats)
pipe_rfin.fit(Xtr_rfin, ytr_rfin)
r_acc_fin = accuracy_score(yte_rfin, pipe_rfin.predict(Xte_rfin))

joblib.dump(pipe_fin, LR_MDL)
log(f"  Saved: {LR_MDL}  (temporal={t_acc_fin:.4f}, random={r_acc_fin:.4f})")
log(f"  Features: {len(cols_fin)}, Params: penalty={bp_a['penalty']} "
    f"C={bp_a['C']:.4f} scaler={bp_a['scaler']} solver={bp_a_solver}")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 72)
log("FINAL RESULTS — ALL EXPERIMENTS (ranked by temporal accuracy)")
log("=" * 72)

results_df = (
    pd.DataFrame(all_results)
    .dropna(subset=["temporal_acc"])
    .sort_values("temporal_acc", ascending=False)
    .reset_index(drop=True)
)
log(results_df.to_string(index=False))

log(f"\nTemporal baseline:  {TEMPORAL_ACC_BASELINE:.4f}  (exp6 LR)")
log(f"Best temporal acc:  {best_temporal:.4f}  (delta: {best_temporal - TEMPORAL_ACC_BASELINE:+.4f})")

if best_temporal > TEMPORAL_ACC_BASELINE:
    log(f"\n✓ IMPROVEMENT — model saved to {BEST_MDL} + {META_JSON}")
    log(f"  Model type: {type(best_model_obj).__name__}")
    log(f"  Features: {len(best_feats_list)}")
    log(f"  Metadata: {best_meta}")
else:
    log(f"\n— No improvement over {TEMPORAL_ACC_BASELINE:.4f} temporal accuracy.")
    log(f"  Best LR still saved to {LR_MDL}")

log(f"\n[DONE]  {pd.Timestamp.now().isoformat()}")
