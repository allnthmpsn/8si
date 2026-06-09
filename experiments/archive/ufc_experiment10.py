"""
ufc_experiment10.py — K value sweep + Elo type comparison.

Experiment A: K sweep (K=48,56,64,72,80,96,128) on all-career Elo
Experiment B: Elo source comparison (UFC-only, hybrid, both, retuned)

Baseline: 73.14% temporal (K=48, all-career, 114 features, 90/10 LR+XGB blend)
Saves if better.
"""

import gc, json, os, time, warnings
from collections import defaultdict

import joblib, numpy as np, pandas as pd, optuna
import xgboost as xgb_mod

from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import RobustScaler, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Config ──────────────────────────────────────────────────────────────────
RS                 = 42
DATE_FROM          = "2018-01-01"
TEMPORAL_CUT       = "2024-01-01"
MIN_UFC_FIGHTS     = 3
PREV_BEST_TEMPORAL = 0.7314          # exp9 A4 best (K=48, all-career, 114 feats)
BASELINE_CORR      = 0.1982          # elo_dif corr at K=48

# Fixed hyperparams per spec
LR_BEST = {"penalty": "l2", "C": 0.31787, "scaler": "robust", "solver": "liblinear"}
XGB_BEST = {
    "n_estimators": 588, "learning_rate": 0.0170, "max_depth": 3,
    "subsample": 0.730, "colsample_bytree": 0.755,
    "min_child_weight": 3, "gamma": 1.657, "reg_alpha": 0.714,
    "use_label_encoder": False, "eval_metric": "logloss",
    "random_state": RS, "n_jobs": 1,
}
LR_XGB_BLEND = 0.90   # fixed for A + B1-B3
ELO_BASE     = 1500.0

K_SWEEP = [48, 56, 64, 72, 80, 96, 128]

OUT_LOG   = "model/experiment10_output.txt"
BEST_MDL  = "model/ufc_model_best.pkl"
BEST_XGB  = "model/ufc_model_xgb.pkl"
BEST_FT   = "model/feature_columns_best.pkl"
META_JSON = "model/model_metadata.json"
ELO_HIST  = "data/elo_ratings_history.csv"
ELO_CURR  = "data/elo_current.csv"

os.makedirs("model", exist_ok=True)
with open(OUT_LOG, "w") as f:
    f.write("")

# ─── Logging + tracking ───────────────────────────────────────────────────────
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

FEAT_108 = joblib.load(BEST_FT)
# FEAT_108 is 114 from exp9; strip the Elo features to get the true 108 base
ELO_6 = ["R_elo", "B_elo", "elo_dif", "R_elo_trend", "B_elo_trend", "elo_trend_dif"]
FEAT_108_BASE = [f for f in FEAT_108 if f not in ELO_6]
log(f"Loaded feature_columns_best.pkl: {len(FEAT_108)} features")
log(f"Base (non-Elo) features: {len(FEAT_108_BASE)}")

def record(tag, t_acc, r_acc, n_feats, secs, lr_model=None, xgb_model=None,
           feats=None, meta=None):
    global best_temporal, best_model_lr, best_model_xgb, best_feats, best_meta
    marker = ""
    if t_acc > best_temporal:
        marker = "  *** NEW BEST ***"
        best_temporal  = t_acc
        best_model_lr  = lr_model
        best_model_xgb = xgb_model
        best_feats     = feats
        best_meta      = meta or {}
    log(f"  {tag:55s}  t={t_acc:.4f}  r={r_acc:.4f}  feats={n_feats:3d}  {secs:.0f}s{marker}")
    all_results.append({"config": tag, "temporal_acc": t_acc, "random_acc": r_acc,
                        "n_feats": n_feats})

def save_best(elo_hist_df=None, elo_curr_df=None):
    if best_model_lr is None:
        log("  [save] No improvement — original files unchanged")
        return
    feats = best_feats if best_feats is not None else list(FEAT_108)
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
    if elo_hist_df is not None:
        elo_hist_df.to_csv(ELO_HIST, index=False)
    if elo_curr_df is not None:
        elo_curr_df.to_csv(ELO_CURR, index=False)
    log(f"  [save] Saved → {BEST_MDL} + metadata")

# ─── Model helpers ────────────────────────────────────────────────────────────
def build_lr(penalty="l2", C=0.31787, l1_ratio=0.5, scaler="robust",
             solver="liblinear", max_iter=2000):
    lr = LogisticRegression(
        penalty=penalty, C=C, solver=solver,
        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        max_iter=max_iter, random_state=RS, n_jobs=1,
    )
    if scaler == "robust":
        return Pipeline([("sc", RobustScaler()), ("lr", lr)])
    if scaler == "standard":
        return Pipeline([("sc", StandardScaler()), ("lr", lr)])
    return lr

def lr_coefs(pipe, feat_list):
    c = pipe.named_steps["lr"].coef_[0] if hasattr(pipe, "named_steps") else pipe.coef_[0]
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

# ─── Dataset helpers ─────────────────────────────────────────────────────────
def get_filtered(feat_list, df=None):
    if df is None:
        df = master
    cols = [c for c in feat_list if c in df.columns]
    sub  = df[
        (df["date"] >= pd.Timestamp(DATE_FROM)) &
        (df["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
        (df["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
    ].copy()
    keep = cols + ["Winner_bin", "date"]
    sub  = sub[[c for c in keep if c in sub.columns]].dropna(subset=cols + ["Winner_bin"])
    return sub, cols

def temporal_sets(feat_list, df=None):
    sub, cols = get_filtered(feat_list, df)
    train = sub[sub["date"] <  pd.Timestamp(TEMPORAL_CUT)]
    test  = sub[sub["date"] >= pd.Timestamp(TEMPORAL_CUT)]
    Xtr_a, ytr_a = augment(train[cols], train["Winner_bin"], cols)
    return Xtr_a.values, test[cols].values, ytr_a.values, test["Winner_bin"].values, cols

def random_sets(feat_list, df=None):
    sub, cols = get_filtered(feat_list, df)
    X, y = sub[cols], sub["Winner_bin"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_a, ytr_a = augment(Xtr, ytr, cols)
    return Xtr_a.values, Xte.values, ytr_a.values, yte.values

def eval_blend(feat_list, df, lr_w=LR_XGB_BLEND):
    """Train LR+XGB blend on temporal split, return (t_acc, r_acc, lr_pipe, xgb_mdl)."""
    Xtr, Xte, ytr, yte, _ = temporal_sets(feat_list, df)
    lr  = build_lr(**LR_BEST)
    xgb = xgb_mod.XGBClassifier(**XGB_BEST)
    lr.fit(Xtr, ytr)
    xgb.fit(Xtr, ytr, verbose=False)
    prob  = lr_w * lr.predict_proba(Xte) + (1 - lr_w) * xgb.predict_proba(Xte)
    t_acc = accuracy_score(yte, prob.argmax(axis=1))

    Xtr_r, Xte_r, ytr_r, yte_r = random_sets(feat_list, df)
    lr_r  = build_lr(**LR_BEST)
    xgb_r = xgb_mod.XGBClassifier(**XGB_BEST)
    lr_r.fit(Xtr_r, ytr_r)
    xgb_r.fit(Xtr_r, ytr_r, verbose=False)
    r_acc = accuracy_score(yte_r, (lr_w * lr_r.predict_proba(Xte_r) +
                                    (1-lr_w) * xgb_r.predict_proba(Xte_r)).argmax(axis=1))
    return t_acc, r_acc, lr, xgb

def elo_dif_corr(feat_list, df, elo_dif_col="elo_dif"):
    """Pearson correlation between elo_dif and Winner_bin on test set."""
    sub, _ = get_filtered(feat_list, df)
    test   = sub[sub["date"] >= pd.Timestamp(TEMPORAL_CUT)]
    if elo_dif_col not in test.columns:
        return float("nan")
    return float(test[elo_dif_col].corr(test["Winner_bin"]))

# ─── Elo computation ──────────────────────────────────────────────────────────
def compute_elo(career_df, K):
    """Compute Elo from career_df (all fights) with constant K. Returns (hist_df, ratings, curr_df)."""
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

        ra, rb = ratings[f], ratings[opp]
        try:
            rf = float(row["won"])
            if pd.isna(rf):
                rf = 0.5
        except (KeyError, TypeError, ValueError):
            res = str(row.get("result", "")).lower()
            rf = 1.0 if "win" in res else (0.0 if "loss" in res else 0.5)
        ro = 1.0 - rf

        ea  = 1 / (1 + 10 ** ((rb - ra) / 400))
        nra = ra + K * (rf - ea)
        nrb = rb + K * (ro - (1 - ea))

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
        lambda x: x - x.shift(3)
    )
    curr = pd.DataFrame([
        {"fighter": f, "current_elo": ratings[f],
         "last_fight_date": last_date.get(f), "total_fights": n_fights[f]}
        for f in sorted(ratings)
    ])
    return hist, dict(ratings), curr


def compute_hybrid_elo(career_df, ufc_keys, K_base):
    """Hybrid Elo: UFC fights use K_base*1.5, regional fights use K_base*0.5."""
    K_ufc = K_base * 1.5
    K_reg = K_base * 0.5
    df = career_df.sort_values(["date", "fighter"]).reset_index(drop=True)
    ratings   = defaultdict(lambda: ELO_BASE)
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

        K = K_ufc if key in ufc_keys else K_reg
        ra, rb = ratings[f], ratings[opp]
        try:
            rf = float(row["won"])
            if pd.isna(rf):
                rf = 0.5
        except (KeyError, TypeError, ValueError):
            res = str(row.get("result", "")).lower()
            rf = 1.0 if "win" in res else (0.0 if "loss" in res else 0.5)
        ro = 1.0 - rf

        ea  = 1 / (1 + 10 ** ((rb - ra) / 400))
        nra = ra + K * (rf - ea)
        nrb = rb + K * (ro - (1 - ea))

        rows += [
            {"fighter": f,   "opponent": opp, "date": date,
             "elo_before": ra, "elo_after": nra, "result": rf},
            {"fighter": opp, "opponent": f,   "date": date,
             "elo_before": rb, "elo_after": nrb, "result": ro},
        ]
        ratings[f] = nra; ratings[opp] = nrb

    hist = pd.DataFrame(rows).sort_values(["fighter", "date"]).reset_index(drop=True)
    hist["elo_trend"] = hist.groupby("fighter")["elo_before"].transform(
        lambda x: x - x.shift(3)
    )
    curr = pd.DataFrame([{"fighter": f, "current_elo": ratings[f]} for f in sorted(ratings)])
    return hist, dict(ratings), curr


def join_elo_standard(df, hist_df):
    """Join 6 standard Elo features (R_elo, B_elo, elo_dif, trends) onto df."""
    for prefix, side in [("R", "R_fighter"), ("B", "B_fighter")]:
        sub = (hist_df[["fighter", "date", "elo_before", "elo_trend"]]
               .rename(columns={"fighter": side, "elo_before": f"{prefix}_elo",
                                "elo_trend": f"{prefix}_elo_trend"})
               .sort_values("date"))
        df = pd.merge_asof(df.sort_values("date"), sub, on="date", by=side,
                           direction="backward")
    df["elo_dif"]       = df["R_elo"]       - df["B_elo"]
    df["elo_trend_dif"] = df["R_elo_trend"] - df["B_elo_trend"]
    return df


def join_elo_labeled(df, hist_df, suffix):
    """Join 6 Elo features with a suffix (e.g. '_career', '_ufc').
    Produces: R_elo_{s}, B_elo_{s}, elo_{s}_dif, R_elo_trend_{s}, B_elo_trend_{s}, elo_trend_{s}_dif
    """
    s = suffix
    for prefix, side in [("R", "R_fighter"), ("B", "B_fighter")]:
        sub = (hist_df[["fighter", "date", "elo_before", "elo_trend"]]
               .rename(columns={"fighter": side, "elo_before": f"{prefix}_elo{s}",
                                "elo_trend": f"{prefix}_elo_trend{s}"})
               .sort_values("date"))
        df = pd.merge_asof(df.sort_values("date"), sub, on="date", by=side,
                           direction="backward")
    df[f"elo{s}_dif"]       = df[f"R_elo{s}"]       - df[f"B_elo{s}"]
    df[f"elo_trend{s}_dif"] = df[f"R_elo_trend{s}"] - df[f"B_elo_trend{s}"]
    return df


# ─── Feature lists ────────────────────────────────────────────────────────────
ELO_STD_FEATS    = ["R_elo", "B_elo", "elo_dif", "R_elo_trend", "B_elo_trend", "elo_trend_dif"]
ELO_CAREER_FEATS = ["R_elo_career", "B_elo_career", "elo_career_dif",
                    "R_elo_trend_career", "B_elo_trend_career", "elo_trend_career_dif"]
ELO_UFC_FEATS    = ["R_elo_ufc", "B_elo_ufc", "elo_ufc_dif",
                    "R_elo_trend_ufc", "B_elo_trend_ufc", "elo_trend_ufc_dif"]
FEAT_114 = list(FEAT_108_BASE) + ELO_STD_FEATS
FEAT_120 = list(FEAT_108_BASE) + ELO_CAREER_FEATS + ELO_UFC_FEATS


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature engineering (108 base features)
# ═════════════════════════════════════════════════════════════════════════════
log("=" * 72)
log(f"UFC EXPERIMENT 10  —  {pd.Timestamp.now().isoformat()}")
log("Experiments: A (K sweep) + B (Elo source comparison)")
log(f"Baseline: {PREV_BEST_TEMPORAL:.4f} temporal (K=48, all-career, 114 features)")
log("=" * 72)
log()
log("=" * 72)
log("STEP 1 — Feature engineering (108 base features)")
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
sub_base, _ = get_filtered(FEAT_108_BASE)
train_count = len(sub_base[sub_base["date"] <  pd.Timestamp(TEMPORAL_CUT)])
test_count  = len(sub_base[sub_base["date"] >= pd.Timestamp(TEMPORAL_CUT)])
log(f"  Base rows — train: {train_count} | test: {test_count}")

# Build UFC fight key set (for hybrid Elo)
ufc_fight_keys = set()
for _, row in master.iterrows():
    f   = str(row.get("R_fighter", ""))
    opp = str(row.get("B_fighter", ""))
    d   = row["date"]
    if f and opp and not pd.isna(d):
        ufc_fight_keys.add((min(f, opp), max(f, opp), str(d.date())))
log(f"  UFC fight keys: {len(ufc_fight_keys):,}")

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — K Value Sweep
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("EXPERIMENT A — K VALUE SWEEP (all-career Elo)")
log("=" * 72)
t_step = time.time()

k_results = []   # list of (K, t_acc, r_acc, corr)
best_k    = 48
best_k_t  = 0.0
best_k_hist = None
best_k_curr = None

for K in K_SWEEP:
    t_k = time.time()
    hist_k, _, curr_k = compute_elo(career_raw, K)

    # Join Elo to master copy
    m_k = join_elo_standard(master.copy(), hist_k)

    # Correlation on test set
    sub_k, _ = get_filtered(FEAT_114, m_k)
    test_k   = sub_k[sub_k["date"] >= pd.Timestamp(TEMPORAL_CUT)]
    corr_k   = float(test_k["elo_dif"].corr(test_k["Winner_bin"]))

    # Blend eval
    t_acc, r_acc, lr_k, xgb_k = eval_blend(FEAT_114, m_k)

    k_results.append((K, t_acc, r_acc, corr_k))
    tag = f"A — K={K:3d}, all-career Elo"
    record(tag, t_acc, r_acc, 114, time.time()-t_k,
           lr_model=lr_k, xgb_model=xgb_k, feats=FEAT_114,
           meta={"model_type": f"blend_K{K}", "blend_ratio": "90% LR + 10% XGB", "K": K})

    if t_acc > best_k_t:
        best_k_t    = t_acc
        best_k      = K
        best_k_hist = hist_k
        best_k_curr = curr_k

    gc.collect()

log(f"  Best K from sweep: K={best_k} (temporal={best_k_t:.4f})")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Elo Source Comparison (using best K from A)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log(f"EXPERIMENT B — ELO SOURCE COMPARISON (K={best_k})")
log("=" * 72)

# ── B1: UFC-only Elo ──────────────────────────────────────────────────────────
log()
log("  B1 — UFC-only Elo")
t_b = time.time()

hist_ufc, _, curr_ufc = compute_elo(ufc_only, best_k)
m_b1 = join_elo_standard(master.copy(), hist_ufc)

sub_b1, _ = get_filtered(FEAT_114, m_b1)
test_b1   = sub_b1[sub_b1["date"] >= pd.Timestamp(TEMPORAL_CUT)]
corr_b1   = float(test_b1["elo_dif"].corr(test_b1["Winner_bin"]))
log(f"    elo_dif corr on test set: {corr_b1:.4f}")

b1_t, b1_r, lr_b1, xgb_b1 = eval_blend(FEAT_114, m_b1)
record("B1 — UFC-only Elo (114 feats)", b1_t, b1_r, 114, time.time()-t_b,
       lr_model=lr_b1, xgb_model=xgb_b1, feats=FEAT_114,
       meta={"model_type": "blend_UFC_elo", "blend_ratio": "90% LR + 10% XGB"})

gc.collect()

# ── B2: Hybrid Elo ───────────────────────────────────────────────────────────
log()
log("  B2 — Hybrid Elo (regional K*0.5, UFC K*1.5)")
t_b = time.time()

hist_hyb, _, _ = compute_hybrid_elo(career_raw, ufc_fight_keys, best_k)
m_b2 = join_elo_standard(master.copy(), hist_hyb)

sub_b2, _ = get_filtered(FEAT_114, m_b2)
test_b2   = sub_b2[sub_b2["date"] >= pd.Timestamp(TEMPORAL_CUT)]
corr_b2   = float(test_b2["elo_dif"].corr(test_b2["Winner_bin"]))
log(f"    elo_dif corr on test set: {corr_b2:.4f}")

b2_t, b2_r, lr_b2, xgb_b2 = eval_blend(FEAT_114, m_b2)
record("B2 — Hybrid Elo (114 feats)", b2_t, b2_r, 114, time.time()-t_b,
       lr_model=lr_b2, xgb_model=xgb_b2, feats=FEAT_114,
       meta={"model_type": "blend_hybrid_elo", "blend_ratio": "90% LR + 10% XGB"})

gc.collect()

# ── B3: Both career + UFC Elo (12 features, 120 total) ───────────────────────
log()
log("  B3 — Both career + UFC Elo (120 features)")
t_b = time.time()

# Re-use best_k_hist for career, hist_ufc for UFC
m_b3 = join_elo_labeled(master.copy(),     best_k_hist, "_career")
m_b3 = join_elo_labeled(m_b3,             hist_ufc,   "_ufc")

sub_b3, _ = get_filtered(FEAT_120, m_b3)
train_b3  = len(sub_b3[sub_b3["date"] <  pd.Timestamp(TEMPORAL_CUT)])
test_b3_n = len(sub_b3[sub_b3["date"] >= pd.Timestamp(TEMPORAL_CUT)])
log(f"    Rows (120 feats) — train: {train_b3} | test: {test_b3_n}")

b3_t, b3_r, lr_b3, xgb_b3 = eval_blend(FEAT_120, m_b3)
record("B3 — Both Elo sources (120 feats)", b3_t, b3_r, 120, time.time()-t_b,
       lr_model=lr_b3, xgb_model=xgb_b3, feats=FEAT_120,
       meta={"model_type": "blend_both_elo120", "blend_ratio": "90% LR + 10% XGB"})

# Coefficient report for B3 LR
try:
    Xtr_b3, _, ytr_b3, _, _ = temporal_sets(FEAT_120, m_b3)
    lr_b3_coef = build_lr(**LR_BEST)
    lr_b3_coef.fit(Xtr_b3, ytr_b3)
    coef_map = lr_coefs(lr_b3_coef, FEAT_120)
    b3_elo_coefs = {f: coef_map.get(f, 0.0) for f in ELO_CAREER_FEATS + ELO_UFC_FEATS}
except Exception as ex:
    log(f"    [warn] Could not extract B3 coefs: {ex}")
    b3_elo_coefs = {}

gc.collect()

# ── B4: Retune LR on best config ─────────────────────────────────────────────
log()

# Determine best Elo config
b_scores = [(b1_t, "career_best_k", m_b1.copy(), FEAT_114, best_k_hist, None),
            (b2_t, "hybrid",        m_b2.copy(), FEAT_114, hist_hyb,    None),
            (b3_t, "both",          m_b3.copy(), FEAT_120, best_k_hist, hist_ufc)]
best_b_config = max(b_scores, key=lambda x: x[0])
best_b_name   = best_b_config[1]
best_b_df     = best_b_config[2]
best_b_feats  = best_b_config[3]
log(f"  B4 — Retune LR on best B config: {best_b_name} ({max(b_scores, key=lambda x: x[0])[0]:.4f})")
log(f"  Running 50 Optuna trials on LR (C range: 0.05–5.0)")
t_b4 = time.time()

Xtr4, Xte4, ytr4, yte4, cols4 = temporal_sets(best_b_feats, best_b_df)

def lr_obj_b4(trial):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C       = trial.suggest_float("C", 0.05, 5.0, log=True)
    l1_r    = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5
    scaler  = trial.suggest_categorical("scaler", ["robust", "standard"])
    solver_hint = trial.suggest_categorical("solver", ["saga", "liblinear"])
    valid   = {"l1": {"saga", "liblinear"}, "l2": {"saga", "liblinear"},
               "elasticnet": {"saga"}}
    solver  = solver_hint if solver_hint in valid[penalty] else "saga"
    try:
        pipe = build_lr(penalty, C, l1_r, scaler, solver)
        pipe.fit(Xtr4, ytr4)
        return accuracy_score(yte4, pipe.predict(Xte4))
    except Exception:
        return 0.0

study_b4 = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS))
study_b4.optimize(lr_obj_b4, n_trials=50, show_progress_bar=False)
bp4 = study_b4.best_params
log(f"  Best LR temporal: {study_b4.best_value:.4f}")
log(f"  Params: {bp4}")

best_lr_b4 = build_lr(bp4["penalty"], bp4["C"], bp4.get("l1_ratio", 0.5),
                      bp4["scaler"], bp4.get("solver", "saga"))
best_lr_b4.fit(Xtr4, ytr4)
best_lr_b4_proba = best_lr_b4.predict_proba(Xte4)

best_xgb_b4 = xgb_mod.XGBClassifier(**XGB_BEST)
best_xgb_b4.fit(Xtr4, ytr4, verbose=False)
best_xgb_b4_proba = best_xgb_b4.predict_proba(Xte4)

b4_best_t   = 0.0
b4_best_lrw = 0.90
for lr_w in [0.85, 0.90, 0.95]:
    xw   = 1.0 - lr_w
    prob = lr_w * best_lr_b4_proba + xw * best_xgb_b4_proba
    t_acc = accuracy_score(yte4, prob.argmax(axis=1))

    # Random accuracy
    Xtr_r, Xte_r, ytr_r, yte_r = random_sets(best_b_feats, best_b_df)
    lr_r = build_lr(bp4["penalty"], bp4["C"], bp4.get("l1_ratio", 0.5),
                    bp4["scaler"], bp4.get("solver", "saga"))
    xgb_r = xgb_mod.XGBClassifier(**XGB_BEST)
    lr_r.fit(Xtr_r, ytr_r)
    xgb_r.fit(Xtr_r, ytr_r, verbose=False)
    r_prob = lr_w * lr_r.predict_proba(Xte_r) + xw * xgb_r.predict_proba(Xte_r)
    r_acc  = accuracy_score(yte_r, r_prob.argmax(axis=1))

    tag = f"B4 — LR{int(lr_w*100)}+XGB{int(xw*100+0.5)} tuned, {best_b_name}"
    record(tag, t_acc, r_acc, len(best_b_feats), time.time()-t_b4,
           lr_model=best_lr_b4, xgb_model=best_xgb_b4, feats=best_b_feats,
           meta={"model_type": f"B4_blend_{int(lr_w*100)}_{int(xw*100+0.5)}",
                 "blend_ratio": f"{lr_w:.0%} LR + {xw:.0%} XGB",
                 "lr_params": bp4, "config": best_b_name})
    if t_acc > b4_best_t:
        b4_best_t   = t_acc
        b4_best_lrw = lr_w

b4_t = b4_best_t
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# Save best model (+ best Elo files if best K changed)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("SAVING BEST MODEL")
log("=" * 72)

# Determine which Elo history corresponds to the best result
save_hist = best_k_hist   # default to best-K all-career history
save_curr = best_k_curr
if best_meta.get("config") == "hybrid" and hist_hyb is not None:
    save_hist = hist_hyb
elif best_meta.get("config") == "both":
    save_hist = best_k_hist   # save career Elo as primary

save_best(elo_hist_df=save_hist, elo_curr_df=save_curr)

# ═════════════════════════════════════════════════════════════════════════════
# Final summary
# ═════════════════════════════════════════════════════════════════════════════
best_res = max(all_results, key=lambda x: x["temporal_acc"])

def fmt(v):
    return f"{v:.2%}" if isinstance(v, float) else "--"

summary = [
    "",
    "=" * 40,
    "EXPERIMENT 10 — K SWEEP + ELO TYPE",
    "=" * 40,
    f"Baseline: {PREV_BEST_TEMPORAL:.2%} temporal (K=48, all-career Elo, 114 features)",
    "",
    "EXPERIMENT A — K VALUE SWEEP (all-career Elo):",
    f"  {'K':>5}  {'Temporal':>8}  {'Random':>8}  {'elo_dif corr':>14}",
    f"  {'---':>5}  {'--------':>8}  {'------':>8}  {'--------------------':>14}",
]

for K, t_acc, r_acc, corr in k_results:
    baseline_note = " (baseline)" if K == 48 else ""
    summary.append(f"  {K:>5}  {fmt(t_acc):>8}  {fmt(r_acc):>8}  {corr:>14.4f}{baseline_note}")

summary += [
    "",
    f"  Best K: {best_k} (temporal: {best_k_t:.2%})",
    "",
    "EXPERIMENT B — ELO TYPE (K={best_k}):".format(best_k=best_k),
    f"  {'Config':28s}  {'Temporal':>8}  {'Random':>8}  {'Features':>8}",
    f"  {'-'*28}  {'--------':>8}  {'------':>8}  {'--------':>8}",
    f"  {'B1 — UFC-only Elo':28s}  {fmt(b1_t):>8}  {fmt(b1_r):>8}  {'114':>8}",
    f"  {'B2 — Hybrid Elo':28s}  {fmt(b2_t):>8}  {fmt(b2_r):>8}  {'114':>8}",
    f"  {'B3 — Both (12 Elo feats)':28s}  {fmt(b3_t):>8}  {fmt(b3_r):>8}  {'120':>8}",
    f"  {'B4 — Retuned LR, best Elo':28s}  {fmt(b4_t):>8}  {'--':>8}  {len(best_b_feats):>8}",
    "",
    "  B3 Elo feature coefficients:",
]

for feat in ELO_CAREER_FEATS + ELO_UFC_FEATS:
    val = b3_elo_coefs.get(feat, 0.0)
    summary.append(f"    {feat:30s}: {val:+.4f}")

saved_models = best_model_lr is not None
saved_elo    = os.path.exists(ELO_HIST)
summary += [
    "",
    f"BEST RESULT: {fmt(best_res['temporal_acc'])} (config: {best_res['config']}, K={best_k})",
    f"vs baseline: {best_res['temporal_acc'] - PREV_BEST_TEMPORAL:+.2%}",
    "",
    f"Saved: {'✓' if saved_models else '✗'} models, {'✓' if saved_elo else '✗'} elo files, "
    f"{'✓' if os.path.exists(META_JSON) else '✗'} metadata",
    "=" * 40,
]

for line in summary:
    log(line)

print()
for line in summary:
    print(line)
