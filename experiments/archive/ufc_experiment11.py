"""
ufc_experiment11.py — Joint retune + Elo variants.

A: Retune LR (100 Optuna) + XGB (75 Optuna) on 114 features, try blend ratios.
B: Dynamic-K Elo (opponent UFC experience scales K).
C: Elo peak features (6 new momentum features → 120 total).

Baseline: 73.14% temporal (K=48, all-career, 114 feats, 90/10 LR+XGB)
"""

import bisect, gc, json, os, time, warnings
from collections import defaultdict

import joblib, numpy as np, pandas as pd, optuna
import xgboost as xgb_mod

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import RobustScaler, StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import accuracy_score
from sklearn.model_selection  import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Config ───────────────────────────────────────────────────────────────────
RS                 = 42
DATE_FROM          = "2018-01-01"
TEMPORAL_CUT       = "2024-01-01"
MIN_UFC_FIGHTS     = 3
PREV_BEST_TEMPORAL = 0.7314
ELO_BASE           = 1500.0
K_BASE             = 48

OUT_LOG   = "model/experiment11_output.txt"
BEST_MDL  = "model/ufc_model_best.pkl"
BEST_XGB  = "model/ufc_model_xgb.pkl"
BEST_FT   = "model/feature_columns_best.pkl"
META_JSON = "model/model_metadata.json"
ELO_HIST  = "data/elo_ratings_history.csv"
ELO_CURR  = "data/elo_current.csv"

os.makedirs("model", exist_ok=True)
with open(OUT_LOG, "w") as f:
    f.write("")

# ─── Logging ──────────────────────────────────────────────────────────────────
def log(msg=""):
    print(msg, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(msg + "\n")

all_results    = []
best_temporal  = PREV_BEST_TEMPORAL
best_model_lr  = None
best_model_xgb = None
best_feats     = None
best_meta      = {}
best_elo_hist  = None
best_elo_curr  = None

# ─── Feature lists ────────────────────────────────────────────────────────────
FEAT_114     = joblib.load(BEST_FT)
ELO_6        = ["R_elo", "B_elo", "elo_dif", "R_elo_trend", "B_elo_trend", "elo_trend_dif"]
FEAT_108_BASE = [f for f in FEAT_114 if f not in ELO_6]
ELO_PEAK_6   = ["R_elo_peak", "B_elo_peak", "R_elo_vs_peak",
                "B_elo_vs_peak", "elo_peak_dif", "elo_vs_peak_dif"]
FEAT_120_PEAK = list(FEAT_114) + ELO_PEAK_6

log(f"Loaded feature_columns_best.pkl: {len(FEAT_114)} features")
log(f"Base (non-Elo) features: {len(FEAT_108_BASE)}")

# ─── Tracking ─────────────────────────────────────────────────────────────────
def record(tag, t_acc, r_acc, n_feats, secs,
           lr_model=None, xgb_model=None, feats=None, meta=None,
           elo_hist=None, elo_curr=None):
    global best_temporal, best_model_lr, best_model_xgb, best_feats
    global best_meta, best_elo_hist, best_elo_curr
    marker = ""
    if t_acc > best_temporal:
        marker          = "  *** NEW BEST ***"
        best_temporal   = t_acc
        best_model_lr   = lr_model
        best_model_xgb  = xgb_model
        best_feats      = feats
        best_meta       = meta or {}
        best_elo_hist   = elo_hist
        best_elo_curr   = elo_curr
    r_str = f"r={r_acc:.4f}" if r_acc is not None else "r=      --"
    log(f"  {tag:55s}  t={t_acc:.4f}  {r_str}  feats={n_feats:3d}  {secs:.0f}s{marker}")
    all_results.append({"config": tag, "temporal_acc": t_acc, "n_feats": n_feats})

def save_best():
    if best_model_lr is None:
        log("  [save] No improvement — original files unchanged")
        return
    feats = best_feats if best_feats is not None else list(FEAT_114)
    joblib.dump(best_model_lr, BEST_MDL)
    joblib.dump(feats, BEST_FT)
    if best_model_xgb is not None:
        joblib.dump(best_model_xgb, BEST_XGB)
    with open(META_JSON, "w") as f:
        json.dump({
            "model_type":        best_meta.get("model_type", "unknown"),
            "temporal_accuracy": best_temporal,
            "n_features":        len(feats),
            "feature_list":      list(feats),
            "blend_ratio":       best_meta.get("blend_ratio", ""),
            "training_window":   f"{DATE_FROM} to <{TEMPORAL_CUT}",
            "date_trained":      pd.Timestamp.now().isoformat(),
        }, f, indent=2)
    if best_elo_hist is not None:
        best_elo_hist.to_csv(ELO_HIST, index=False)
    if best_elo_curr is not None:
        best_elo_curr.to_csv(ELO_CURR, index=False)
    log(f"  [save] Saved → {BEST_MDL} + metadata")

# ─── Model builders ───────────────────────────────────────────────────────────
def build_lr(penalty="l2", C=1.0, l1_ratio=0.5, scaler="robust",
             solver="saga", max_iter=2000):
    lr = LogisticRegression(
        penalty=penalty, C=C, solver=solver,
        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        max_iter=max_iter, random_state=RS, n_jobs=1,
    )
    sc = RobustScaler() if scaler == "robust" else StandardScaler()
    return Pipeline([("sc", sc), ("lr", lr)])

def build_xgb(n_estimators=500, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              gamma=1.0, reg_alpha=1.0, reg_lambda=1.0):
    return xgb_mod.XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=subsample,
        colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RS, n_jobs=1,
    )

def lr_coefs(pipe, feat_list):
    c = pipe.named_steps["lr"].coef_[0]
    return dict(zip(feat_list, c))

# ─── Augmentation ─────────────────────────────────────────────────────────────
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

# ─── Dataset helpers ──────────────────────────────────────────────────────────
def get_filtered(feat_list, df):
    cols = [c for c in feat_list if c in df.columns]
    sub  = df[
        (df["date"] >= pd.Timestamp(DATE_FROM)) &
        (df["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
        (df["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
    ].copy()
    keep = cols + ["Winner_bin", "date"]
    sub  = sub[[c for c in keep if c in sub.columns]].dropna(subset=cols + ["Winner_bin"])
    return sub, cols

def temporal_sets(feat_list, df):
    sub, cols = get_filtered(feat_list, df)
    train = sub[sub["date"] <  pd.Timestamp(TEMPORAL_CUT)]
    test  = sub[sub["date"] >= pd.Timestamp(TEMPORAL_CUT)]
    Xtr_a, ytr_a = augment(train[cols], train["Winner_bin"], cols)
    return Xtr_a.values, test[cols].values, ytr_a.values, test["Winner_bin"].values, cols

def random_sets(feat_list, df):
    sub, cols = get_filtered(feat_list, df)
    X, y = sub[cols], sub["Winner_bin"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_a, ytr_a = augment(Xtr, ytr, cols)
    return Xtr_a.values, Xte.values, ytr_a.values, yte.values

# ─── Elo join helpers ─────────────────────────────────────────────────────────
def join_elo_standard(df, hist_df):
    for prefix, side in [("R", "R_fighter"), ("B", "B_fighter")]:
        sub = (hist_df[["fighter", "date", "elo_before", "elo_trend"]]
               .rename(columns={"fighter": side,
                                "elo_before": f"{prefix}_elo",
                                "elo_trend":  f"{prefix}_elo_trend"})
               .sort_values("date"))
        df = pd.merge_asof(df.sort_values("date"), sub, on="date", by=side,
                           direction="backward")
    df["elo_dif"]       = df["R_elo"]       - df["B_elo"]
    df["elo_trend_dif"] = df["R_elo_trend"] - df["B_elo_trend"]
    return df

def join_elo_with_peak(df, hist_df):
    """Join standard 6 Elo features + 6 peak features (requires elo_peak column in hist_df)."""
    for prefix, side in [("R", "R_fighter"), ("B", "B_fighter")]:
        sub = (hist_df[["fighter", "date", "elo_before", "elo_trend", "elo_peak"]]
               .rename(columns={"fighter": side,
                                "elo_before": f"{prefix}_elo",
                                "elo_trend":  f"{prefix}_elo_trend",
                                "elo_peak":   f"{prefix}_elo_peak"})
               .sort_values("date"))
        df = pd.merge_asof(df.sort_values("date"), sub, on="date", by=side,
                           direction="backward")
    df["elo_dif"]         = df["R_elo"]         - df["B_elo"]
    df["elo_trend_dif"]   = df["R_elo_trend"]   - df["B_elo_trend"]
    df["R_elo_vs_peak"]   = df["R_elo"]         - df["R_elo_peak"]
    df["B_elo_vs_peak"]   = df["B_elo"]         - df["B_elo_peak"]
    df["elo_peak_dif"]    = df["R_elo_peak"]    - df["B_elo_peak"]
    df["elo_vs_peak_dif"] = df["R_elo_vs_peak"] - df["B_elo_vs_peak"]
    return df

# ─── Dynamic-K Elo computation ────────────────────────────────────────────────
def compute_dynamic_elo(career_df, ufc_dates_by_fighter, base_k=48):
    """Elo with K scaled by opponent's UFC fight count at the time of the fight."""
    def get_k(opp, date):
        dates = ufc_dates_by_fighter.get(opp, [])
        n = bisect.bisect_left(dates, date)   # count of UFC dates strictly < date
        if   n == 0:  return base_k * 0.4
        elif n <= 3:  return base_k * 0.7
        elif n <= 8:  return base_k * 1.0
        elif n <= 15: return base_k * 1.3
        else:         return base_k * 1.6

    df = career_df.sort_values(["date", "fighter"]).reset_index(drop=True)
    ratings   = defaultdict(lambda: ELO_BASE)
    last_date = {}
    n_fights  = defaultdict(int)
    processed = set()
    rows      = []

    for _, row in df.iterrows():
        f    = str(row.get("fighter", ""))
        opp  = str(row.get("opponent", ""))
        date = row["date"]
        if not f or not opp or f == opp or pd.isna(date):
            continue
        key = (min(f, opp), max(f, opp), str(date.date()))
        if key in processed:
            continue
        processed.add(key)

        K_f = get_k(opp, date)   # K applied to fighter's update
        K_o = get_k(f,   date)   # K applied to opponent's update
        ra, rb = ratings[f], ratings[opp]

        try:
            rf = float(row["won"])
            if pd.isna(rf): rf = 0.5
        except (KeyError, TypeError, ValueError):
            res = str(row.get("result", "")).lower()
            rf = 1.0 if "win" in res else (0.0 if "loss" in res else 0.5)
        ro = 1.0 - rf

        ea  = 1 / (1 + 10 ** ((rb - ra) / 400))
        nra = ra + K_f * (rf - ea)
        nrb = rb + K_o * (ro - (1 - ea))

        rows += [
            {"fighter": f,   "opponent": opp, "date": date,
             "elo_before": ra, "elo_after": nra, "result": rf},
            {"fighter": opp, "opponent": f,   "date": date,
             "elo_before": rb, "elo_after": nrb, "result": ro},
        ]
        ratings[f]   = nra;  ratings[opp] = nrb
        last_date[f] = date; last_date[opp] = date
        n_fights[f] += 1;    n_fights[opp] += 1

    hist = pd.DataFrame(rows).sort_values(["fighter", "date"]).reset_index(drop=True)
    hist["elo_trend"] = hist.groupby("fighter")["elo_before"].transform(
        lambda x: x - x.shift(3))
    curr = pd.DataFrame([
        {"fighter": f, "current_elo": ratings[f],
         "last_fight_date": last_date.get(f), "total_fights": n_fights[f]}
        for f in sorted(ratings)
    ])
    return hist, dict(ratings), curr

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature engineering (108 base features, identical to exp10)
# ═════════════════════════════════════════════════════════════════════════════
log("=" * 72)
log(f"UFC EXPERIMENT 11  —  {pd.Timestamp.now().isoformat()}")
log("Experiments: A (joint retune) + B (dynamic K Elo) + C (peak features)")
log(f"Baseline: {PREV_BEST_TEMPORAL:.4f} temporal (K=48, all-career, 114 features)")
log("=" * 72)
log()
log("=" * 72)
log("STEP 1 — Feature engineering")
log("=" * 72)
t0 = time.time()

career_raw = pd.read_csv("data/career_fights_updated.csv")
career_raw["date"] = pd.to_datetime(career_raw["date"])
master     = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
fighters   = pd.read_csv("data/ufc_fighters_final_updated.csv")
ufc_only   = pd.read_csv("data/ufc_stats_fights.csv")
ufc_only["date"] = pd.to_datetime(ufc_only["date"])

log(f"  career_fights_updated: {len(career_raw):,} rows")
log(f"  ufc_stats_fights:      {len(ufc_only):,} rows")

career = career_raw.sort_values(["fighter", "date"]).reset_index(drop=True).copy()
g = career.groupby("fighter", sort=False)

career["cum_wins"]        = g["won"].transform(lambda x: x.shift(1).cumsum().fillna(0))
career["cum_fights"]      = g["won"].transform(lambda x: x.shift(1).expanding().count().fillna(0))
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
    career.sort_values("date"), opp_src, on="date", by="opponent", direction="backward")
career_with_opp = career_with_opp.sort_values(["fighter", "date"])
career_with_opp["opp_quality"] = (
    career_with_opp.groupby("fighter")["opp_win_rate"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

JOIN_COLS = [
    "career_win_rate", "ko_finish_rate", "sub_finish_rate",
    "last3_win_rate", "last10_win_rate", "trend_score",
    "layoff_days", "last5_won", "last5_finish_rate",
    "cum_wins", "cum_fights", "career_finish_rate", "recency_win_rate",
]
career_feat = career_with_opp[["fighter", "date"] + JOIN_COLS + ["opp_quality"]].copy()

def join_career(df, prefix):
    fc = f"{prefix}_fighter"
    sub = career_feat.rename(columns={
        "fighter": fc, **{c: f"{prefix}_{c}" for c in JOIN_COLS + ["opp_quality"]}
    }).sort_values("date")
    return pd.merge_asof(df.sort_values("date"), sub, on="date", by=fc, direction="backward")

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

def add_diff(df, col):
    rc, bc = f"R_{col}", f"B_{col}"
    if rc in df.columns and bc in df.columns:
        df[f"{col}_dif"] = df[rc] - df[bc]

for col in [
    "career_win_rate", "ko_finish_rate", "sub_finish_rate",
    "trend_score", "opp_quality", "last3_win_rate", "last10_win_rate",
    "layoff_days", "last5_won", "last5_finish_rate",
    "SLpM", "SApM", "Str_Def", "TD_Def", "Sub_Avg", "TD_Avg",
    "career_finish_rate", "recency_win_rate",
]:
    add_diff(master, col)

for p in ["R", "B"]:
    master[f"{p}_age_x_exp"] = master[f"{p}_age"] * master[f"{p}_cum_fights"]
add_diff(master, "age_x_exp")

for p in ["R", "B"]:
    ld = master[f"{p}_layoff_days"]
    master[f"{p}_layoff_lt90"]    = (ld <  90).astype(int)
    master[f"{p}_layoff_90_180"]  = ((ld >= 90)  & (ld < 180)).astype(int)
    master[f"{p}_layoff_180_365"] = ((ld >= 180) & (ld < 365)).astype(int)
    master[f"{p}_layoff_gt365"]   = (ld >= 365).astype(int)

log(f"  Feature engineering done in {time.time()-t0:.1f}s")
gc.collect()

# ─── Build UFC dates lookup for dynamic K ─────────────────────────────────────
# For each fighter: sorted list of UFC fight dates (for bisect lookup)
ufc_dates_by_fighter = {}
for fighter, grp in ufc_only.groupby("fighter"):
    ufc_dates_by_fighter[fighter] = sorted(grp["date"].tolist())
log(f"  UFC dates lookup: {len(ufc_dates_by_fighter):,} fighters")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load existing K=48 Elo + join to master
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 2 — Load K=48 all-career Elo")
log("=" * 72)
t_elo = time.time()

hist_48 = pd.read_csv(ELO_HIST)
hist_48["date"] = pd.to_datetime(hist_48["date"])
hist_48 = hist_48.sort_values(["fighter", "date"]).reset_index(drop=True)
log(f"  elo_ratings_history: {len(hist_48):,} rows")

# Peak Elo — expanding max of elo_before per fighter (pre-fight peak)
hist_48["elo_peak"] = hist_48.groupby("fighter")["elo_before"].transform(
    lambda x: x.expanding().max()
)

# Join standard 6 Elo features to master (for experiments A and B comparison)
master_elo = join_elo_standard(master.copy(), hist_48)
# Join 12 features (standard + peak) for experiment C
master_peak = join_elo_with_peak(master.copy(), hist_48)

sub_base, _ = get_filtered(FEAT_114, master_elo)
log(f"  master_elo rows — train: {len(sub_base[sub_base['date'] < pd.Timestamp(TEMPORAL_CUT)])} "
    f"| test: {len(sub_base[sub_base['date'] >= pd.Timestamp(TEMPORAL_CUT)])}")
log(f"  Elo load done in {time.time()-t_elo:.1f}s")
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A1 — Retune LR on 114 features (100 Optuna trials)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("EXPERIMENT A1 — LR Retune (100 Optuna trials, 114 features)")
log("=" * 72)
t_a1 = time.time()

Xtr_a, Xte_a, ytr_a, yte_a, cols_a = temporal_sets(FEAT_114, master_elo)
log(f"  Train: {len(Xtr_a)} (aug) | Test: {len(Xte_a)}")

def lr_obj(trial):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C       = trial.suggest_float("C", 0.001, 10.0, log=True)
    l1_r    = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5
    scaler  = trial.suggest_categorical("scaler", ["robust", "standard"])
    solver_hint = trial.suggest_categorical("solver", ["saga", "liblinear"])
    valid   = {"l1": {"saga", "liblinear"}, "l2": {"saga", "liblinear"},
               "elasticnet": {"saga"}}
    solver  = solver_hint if solver_hint in valid[penalty] else "saga"
    try:
        pipe = build_lr(penalty, C, l1_r, scaler, solver)
        pipe.fit(Xtr_a, ytr_a)
        return accuracy_score(yte_a, pipe.predict(Xte_a))
    except Exception:
        return 0.0

study_lr = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=RS))
study_lr.optimize(lr_obj, n_trials=100, show_progress_bar=False)
bp_lr = study_lr.best_params
a1_best_t = study_lr.best_value

log(f"  Best LR temporal: {a1_best_t:.4f}")
log(f"  Params: penalty={bp_lr['penalty']}, C={bp_lr['C']:.5f}, "
    f"l1_ratio={bp_lr.get('l1_ratio', 'N/A')}, "
    f"scaler={bp_lr['scaler']}, solver={bp_lr.get('solver', 'saga')}")

# Resolve solver (same fix as objective)
valid_solvers = {"l1": {"saga", "liblinear"}, "l2": {"saga", "liblinear"},
                 "elasticnet": {"saga"}}
a1_solver = bp_lr.get("solver", "saga")
if a1_solver not in valid_solvers[bp_lr["penalty"]]:
    a1_solver = "saga"

# Build best LR params dict for downstream use
BEST_LR_PARAMS = {
    "penalty":  bp_lr["penalty"],
    "C":        bp_lr["C"],
    "l1_ratio": bp_lr.get("l1_ratio", 0.5),
    "scaler":   bp_lr["scaler"],
    "solver":   a1_solver,
}

log(f"  A1 done in {time.time()-t_a1:.0f}s")
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A2 — Retune XGB on 114 features (75 Optuna trials)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("EXPERIMENT A2 — XGB Retune (75 Optuna trials, 114 features)")
log("=" * 72)
t_a2 = time.time()

def xgb_obj(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth":       trial.suggest_int("max_depth", 2, 5),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":trial.suggest_int("min_child_weight", 1, 15),
        "gamma":           trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda":      trial.suggest_float("reg_lambda", 0.5, 3.0),
    }
    try:
        mdl = build_xgb(**params)
        mdl.fit(Xtr_a, ytr_a, verbose=False)
        return accuracy_score(yte_a, mdl.predict(Xte_a))
    except Exception:
        return 0.0

study_xgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RS))
study_xgb.optimize(xgb_obj, n_trials=75, show_progress_bar=False)
bp_xgb = study_xgb.best_params
a2_best_t = study_xgb.best_value

log(f"  Best XGB temporal: {a2_best_t:.4f}")
log(f"  Params: n_est={bp_xgb['n_estimators']}, lr={bp_xgb['learning_rate']:.4f}, "
    f"depth={bp_xgb['max_depth']}, sub={bp_xgb['subsample']:.3f}, "
    f"col={bp_xgb['colsample_bytree']:.3f}, mcw={bp_xgb['min_child_weight']}, "
    f"gamma={bp_xgb['gamma']:.3f}, reg_alpha={bp_xgb['reg_alpha']:.3f}, "
    f"reg_lambda={bp_xgb['reg_lambda']:.3f}")

BEST_XGB_PARAMS = dict(bp_xgb)
log(f"  A2 done in {time.time()-t_a2:.0f}s")
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A3 — Blend ratios with jointly tuned models
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("EXPERIMENT A3 — Blend ratios (jointly tuned LR + XGB)")
log("=" * 72)
t_a3 = time.time()

# Train final temporal models with best params
lr_a3  = build_lr(**BEST_LR_PARAMS)
xgb_a3 = build_xgb(**BEST_XGB_PARAMS)
lr_a3.fit(Xtr_a, ytr_a)
xgb_a3.fit(Xtr_a, ytr_a, verbose=False)
lr_proba_a3  = lr_a3.predict_proba(Xte_a)
xgb_proba_a3 = xgb_a3.predict_proba(Xte_a)

# Train random-split models
Xtr_r, Xte_r, ytr_r, yte_r = random_sets(FEAT_114, master_elo)
lr_r_a3  = build_lr(**BEST_LR_PARAMS)
xgb_r_a3 = build_xgb(**BEST_XGB_PARAMS)
lr_r_a3.fit(Xtr_r, ytr_r)
xgb_r_a3.fit(Xtr_r, ytr_r, verbose=False)

blend_results = []
for lr_w in [1.00, 0.95, 0.90, 0.85, 0.80]:
    xw    = 1.0 - lr_w
    prob  = lr_w * lr_proba_a3 + xw * xgb_proba_a3
    t_acc = accuracy_score(yte_a, prob.argmax(axis=1))

    r_prob = lr_w * lr_r_a3.predict_proba(Xte_r) + xw * xgb_r_a3.predict_proba(Xte_r)
    r_acc  = accuracy_score(yte_r, r_prob.argmax(axis=1))

    blend_results.append((lr_w, t_acc, r_acc))
    tag = (f"A3 — LR{int(lr_w*100)}+XGB{int(round(xw*100))} (jointly tuned)" if xw > 0
           else "A3 — LR100 (solo, jointly tuned)")
    record(tag, t_acc, r_acc, 114, time.time()-t_a3,
           lr_model=lr_a3, xgb_model=xgb_a3 if xw > 0 else None,
           feats=FEAT_114,
           meta={"model_type": f"A3_blend_LR{int(lr_w*100)}_XGB{int(round(xw*100))}",
                 "blend_ratio": f"{lr_w:.0%} LR + {xw:.0%} XGB",
                 "lr_params": BEST_LR_PARAMS, "xgb_params": BEST_XGB_PARAMS})

best_a3_lrw, best_a3_t, _ = max(blend_results, key=lambda x: x[1])
log(f"  Best blend: LR{int(best_a3_lrw*100)}/XGB{int(round((1-best_a3_lrw)*100))} = {best_a3_t:.4f}")
log(f"  A3 done in {time.time()-t_a3:.0f}s")
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Dynamic-K Elo (opponent UFC experience)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("EXPERIMENT B — Dynamic-K Elo (opponent UFC experience)")
log("=" * 72)
t_b = time.time()

log("  Computing dynamic-K Elo from all career fights...")
hist_dyn, ratings_dyn, curr_dyn = compute_dynamic_elo(career_raw, ufc_dates_by_fighter, K_BASE)
log(f"  Dynamic Elo history: {len(hist_dyn):,} rows")

master_dyn = join_elo_standard(master.copy(), hist_dyn)

# Correlation on test set
sub_dyn, _ = get_filtered(FEAT_114, master_dyn)
test_dyn   = sub_dyn[sub_dyn["date"] >= pd.Timestamp(TEMPORAL_CUT)]
corr_dyn   = float(test_dyn["elo_dif"].corr(test_dyn["Winner_bin"]))
log(f"  elo_dif corr (dynamic K): {corr_dyn:.4f}  vs K=48 fixed: 0.1982")

# Train with best jointly tuned params from A
Xtr_b, Xte_b, ytr_b, yte_b, _ = temporal_sets(FEAT_114, master_dyn)
lr_b  = build_lr(**BEST_LR_PARAMS)
xgb_b = build_xgb(**BEST_XGB_PARAMS)
lr_b.fit(Xtr_b, ytr_b)
xgb_b.fit(Xtr_b, ytr_b, verbose=False)

lr_proba_b  = lr_b.predict_proba(Xte_b)
xgb_proba_b = xgb_b.predict_proba(Xte_b)

# Try same best blend ratio from A3
prob_b = best_a3_lrw * lr_proba_b + (1 - best_a3_lrw) * xgb_proba_b
b_t    = accuracy_score(yte_b, prob_b.argmax(axis=1))

Xtr_br, Xte_br, ytr_br, yte_br = random_sets(FEAT_114, master_dyn)
lr_br  = build_lr(**BEST_LR_PARAMS); xgb_br = build_xgb(**BEST_XGB_PARAMS)
lr_br.fit(Xtr_br, ytr_br); xgb_br.fit(Xtr_br, ytr_br, verbose=False)
b_r = accuracy_score(yte_br,
    (best_a3_lrw * lr_br.predict_proba(Xte_br) +
     (1-best_a3_lrw) * xgb_br.predict_proba(Xte_br)).argmax(axis=1))

# elo_dif coefficient from LR
coef_dyn_elo = lr_coefs(lr_b, FEAT_114).get("elo_dif", float("nan"))

record("B — Dynamic-K Elo (opp UFC exp, 114 feats)", b_t, b_r, 114, time.time()-t_b,
       lr_model=lr_b, xgb_model=xgb_b, feats=FEAT_114,
       meta={"model_type": "B_dynamic_K_elo",
             "blend_ratio": f"{best_a3_lrw:.0%} LR + {1-best_a3_lrw:.0%} XGB"},
       elo_hist=hist_dyn, elo_curr=curr_dyn)

log(f"  elo_dif coefficient: {coef_dyn_elo:+.4f}  (K=48 baseline: +0.2265)")
log(f"  B done in {time.time()-t_b:.0f}s")
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Elo peak features (120 total features)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("EXPERIMENT C — Elo peak features (6 new, 120 total)")
log("=" * 72)
t_c = time.time()

sub_c, _ = get_filtered(FEAT_120_PEAK, master_peak)
log(f"  Rows (120 feats) — train: {len(sub_c[sub_c['date'] < pd.Timestamp(TEMPORAL_CUT)])} "
    f"| test: {len(sub_c[sub_c['date'] >= pd.Timestamp(TEMPORAL_CUT)])}")

Xtr_c, Xte_c, ytr_c, yte_c, _ = temporal_sets(FEAT_120_PEAK, master_peak)
lr_c  = build_lr(**BEST_LR_PARAMS)
xgb_c = build_xgb(**BEST_XGB_PARAMS)
lr_c.fit(Xtr_c, ytr_c)
xgb_c.fit(Xtr_c, ytr_c, verbose=False)

lr_proba_c  = lr_c.predict_proba(Xte_c)
xgb_proba_c = xgb_c.predict_proba(Xte_c)

prob_c = best_a3_lrw * lr_proba_c + (1 - best_a3_lrw) * xgb_proba_c
c_t    = accuracy_score(yte_c, prob_c.argmax(axis=1))

Xtr_cr, Xte_cr, ytr_cr, yte_cr = random_sets(FEAT_120_PEAK, master_peak)
lr_cr  = build_lr(**BEST_LR_PARAMS); xgb_cr = build_xgb(**BEST_XGB_PARAMS)
lr_cr.fit(Xtr_cr, ytr_cr); xgb_cr.fit(Xtr_cr, ytr_cr, verbose=False)
c_r = accuracy_score(yte_cr,
    (best_a3_lrw * lr_cr.predict_proba(Xte_cr) +
     (1-best_a3_lrw) * xgb_cr.predict_proba(Xte_cr)).argmax(axis=1))

# Coefficients for new peak features
coef_c = lr_coefs(lr_c, FEAT_120_PEAK)
peak_coefs = {f: coef_c.get(f, 0.0) for f in ELO_PEAK_6}

record("C — Elo peak features (120 feats)", c_t, c_r, 120, time.time()-t_c,
       lr_model=lr_c, xgb_model=xgb_c, feats=FEAT_120_PEAK,
       meta={"model_type": "C_elo_peak_120",
             "blend_ratio": f"{best_a3_lrw:.0%} LR + {1-best_a3_lrw:.0%} XGB"})

log(f"  C done in {time.time()-t_c:.0f}s")
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("SAVING BEST MODEL")
log("=" * 72)
save_best()

# ═════════════════════════════════════════════════════════════════════════════
# Final summary
# ═════════════════════════════════════════════════════════════════════════════
best_res = max(all_results, key=lambda x: x["temporal_acc"])

def fmt(v):
    return f"{v:.2%}" if isinstance(v, float) else "--"

summary = [
    "",
    "=" * 40,
    "EXPERIMENT 11 — JOINT RETUNE + ELO VARIANTS",
    "=" * 40,
    f"Baseline: {PREV_BEST_TEMPORAL:.2%} temporal",
    "",
    "EXPERIMENT A — JOINT RETUNE (114 features, K=48):",
    f"  Best LR params: penalty={BEST_LR_PARAMS['penalty']}, "
    f"C={BEST_LR_PARAMS['C']:.5f}, "
    f"l1_ratio={BEST_LR_PARAMS['l1_ratio']:.3f}, "
    f"scaler={BEST_LR_PARAMS['scaler']}",
    f"  Best XGB params: n_estimators={BEST_XGB_PARAMS['n_estimators']}, "
    f"lr={BEST_XGB_PARAMS['learning_rate']:.4f}, "
    f"depth={BEST_XGB_PARAMS['max_depth']}, "
    f"gamma={BEST_XGB_PARAMS['gamma']:.3f}, "
    f"reg_alpha={BEST_XGB_PARAMS['reg_alpha']:.3f}",
    "",
    "  Blend ratios:",
]

for lr_w, t_acc, r_acc in blend_results:
    xw    = 1.0 - lr_w
    label = (f"    {int(lr_w*100)}% LR + {int(round(xw*100))}% XGB:" if xw > 0
             else "    100% LR:                ")
    tag   = " (current production)" if abs(lr_w - 0.90) < 0.01 else ""
    summary.append(f"  {label:30s}  {fmt(t_acc)}{tag}")

summary += [
    "",
    f"  Best blend: {int(best_a3_lrw*100)}% LR + {int(round((1-best_a3_lrw)*100))}% XGB"
    f" = {fmt(best_a3_t)} temporal",
    "",
    "EXPERIMENT B — DYNAMIC K ELO:",
    f"  Temporal: {fmt(b_t)} | vs baseline: {b_t - PREV_BEST_TEMPORAL:+.2%}",
    f"  elo_dif coefficient: {coef_dyn_elo:+.4f} (vs +0.2265 at K=48)",
    "",
    "EXPERIMENT C — ELO PEAK FEATURES (120 feats):",
    f"  Temporal: {fmt(c_t)} | vs baseline: {c_t - PREV_BEST_TEMPORAL:+.2%}",
    "  New feature coefficients:",
    f"    elo_peak_dif:     {peak_coefs.get('elo_peak_dif', 0.0):+.4f}",
    f"    elo_vs_peak_dif:  {peak_coefs.get('elo_vs_peak_dif', 0.0):+.4f}",
    "",
    f"BEST RESULT: {fmt(best_res['temporal_acc'])} (config: {best_res['config']})",
    f"vs baseline: {best_res['temporal_acc'] - PREV_BEST_TEMPORAL:+.2%}",
    "",
    f"Saved: {'✓' if best_model_lr is not None else '✗'} models  "
    f"{'✓' if best_elo_hist is not None and best_model_lr is not None else '✗'} elo  "
    f"{'✓' if os.path.exists(META_JSON) and best_model_lr is not None else '✗'} metadata",
    "=" * 40,
]

for line in summary:
    log(line)

print()
for line in summary:
    print(line)
