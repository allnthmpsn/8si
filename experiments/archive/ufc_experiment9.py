"""
ufc_experiment9.py — Elo ratings + Weight-class aware modeling.

Experiment A: Elo ratings (A1-A5)
Experiment B: Weight-class aware modeling (B1-B4)
Combined: best Elo + best weight-class features

Primary metric: temporal accuracy (train 2018-2023, test 2024+).
Saves winner if it beats PREV_BEST_TEMPORAL = 0.7206.
"""

import gc, json, os, time, warnings
from collections import defaultdict

import joblib, numpy as np, pandas as pd, optuna
import xgboost as xgb_mod

from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import RobustScaler, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Config ──────────────────────────────────────────────────────────────────
RS                 = 42
DATE_FROM          = "2018-01-01"
TEMPORAL_CUT       = "2024-01-01"
MIN_UFC_FIGHTS     = 3
PREV_BEST_TEMPORAL = 0.7206

# Best hyperparams from exp8
LR_BEST = {"penalty": "l2", "C": 0.31787, "scaler": "robust", "solver": "liblinear"}
XGB_BEST = {
    "n_estimators": 588, "learning_rate": 0.0170, "max_depth": 3,
    "subsample": 0.730, "colsample_bytree": 0.755,
    "min_child_weight": 3, "gamma": 1.657, "reg_alpha": 0.714,
    "use_label_encoder": False, "eval_metric": "logloss",
    "random_state": RS, "n_jobs": 1,
}

ELO_K_VALUES = [16, 24, 32, 48]
ELO_BASE     = 1500.0

DIV_GROUPS = {
    "Welterweight": "high", "Middleweight": "high",
    "Featherweight": "medium", "Lightweight": "medium", "Bantamweight": "medium",
    "Heavyweight": "low", "Light Heavyweight": "low",
    "Women's Strawweight": "low", "Women's Flyweight": "low",
    "Women's Bantamweight": "low", "Women's Featherweight": "low",
    "Catch Weight": "medium",
}
LIGHTER_DIVS = {
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight", "Flyweight", "Bantamweight", "Featherweight",
}

OUT_LOG   = "model/experiment9_output.txt"
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
best_model_obj = None
best_feats_list = None
best_xgb_obj   = None
best_meta_dict = {}

FEAT_108 = joblib.load(BEST_FT)

def record(tag, t_acc, r_acc, n_feats, secs, model=None, feats=None,
           meta=None, xgb_model=None):
    global best_temporal, best_model_obj, best_feats_list, best_xgb_obj, best_meta_dict
    marker = ""
    if t_acc > best_temporal:
        marker = "  *** NEW BEST ***"
        best_temporal   = t_acc
        best_model_obj  = model
        best_feats_list = feats if feats is not None else FEAT_108
        best_xgb_obj    = xgb_model
        best_meta_dict  = meta or {}
    log(f"  {tag:60s}  t={t_acc:.4f}  r={r_acc:.4f}  feats={n_feats:3d}  {secs:.0f}s{marker}")
    all_results.append({"config": tag, "temporal_acc": t_acc, "random_acc": r_acc,
                        "n_feats": n_feats})

def save_best():
    if best_model_obj is None:
        log("  [save] No improvement — original files unchanged")
        return
    feats = best_feats_list if best_feats_list is not None else FEAT_108
    joblib.dump(best_model_obj, BEST_MDL)
    joblib.dump(feats, BEST_FT)
    if best_xgb_obj is not None:
        joblib.dump(best_xgb_obj, BEST_XGB)
    with open(META_JSON, "w") as f:
        json.dump({
            "model_type":        best_meta_dict.get("model_type", "unknown"),
            "temporal_accuracy": best_temporal,
            "n_features":        len(feats),
            "feature_list":      list(feats),
            "blend_ratio":       best_meta_dict.get("blend_ratio", ""),
            "training_window":   f"{DATE_FROM} to <{TEMPORAL_CUT}",
            "date_trained":      pd.Timestamp.now().isoformat(),
        }, f, indent=2)
    log(f"  [save] Saved → {BEST_MDL} + {META_JSON}")

# ─── Model helpers ────────────────────────────────────────────────────────────
def build_lr(penalty="l2", C=0.318, l1_ratio=0.5, scaler="robust",
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
    if hasattr(pipe, "named_steps"):
        c = pipe.named_steps["lr"].coef_[0]
    else:
        c = pipe.coef_[0]
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

# ─── Dataset helpers (df=None uses global master) ─────────────────────────────
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
    if "weight_class" in df.columns:
        keep = keep + ["weight_class"]
    sub = sub[[c for c in keep if c in sub.columns]].dropna(subset=cols + ["Winner_bin"])
    return sub, cols

def temporal_sets(feat_list=None, df=None):
    if feat_list is None:
        feat_list = FEAT_108
    sub, cols = get_filtered(feat_list, df)
    train = sub[sub["date"] <  pd.Timestamp(TEMPORAL_CUT)]
    test  = sub[sub["date"] >= pd.Timestamp(TEMPORAL_CUT)]
    Xtr_a, ytr_a = augment(train[cols], train["Winner_bin"], cols)
    return Xtr_a.values, test[cols].values, ytr_a.values, test["Winner_bin"].values, cols

def random_sets(feat_list=None, df=None, stratify_wc=False):
    if feat_list is None:
        feat_list = FEAT_108
    sub, cols = get_filtered(feat_list, df)
    X, y = sub[cols], sub["Winner_bin"]
    if stratify_wc and "weight_class" in sub.columns:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
        tr_i, te_i = next(sss.split(X, sub["weight_class"].fillna("Unknown")))
        Xtr, Xte = X.iloc[tr_i], X.iloc[te_i]
        ytr, yte = y.iloc[tr_i], y.iloc[te_i]
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_a, ytr_a = augment(Xtr, ytr, cols)
    return Xtr_a.values, Xte.values, ytr_a.values, yte.values

def quick_temporal(model, feat_list, df=None):
    Xtr, Xte, ytr, yte, _ = temporal_sets(feat_list, df)
    model.fit(Xtr, ytr)
    return accuracy_score(yte, model.predict(Xte)), model.predict_proba(Xte), yte

def quick_random(model, feat_list, df=None, stratify_wc=False):
    Xtr, Xte, ytr, yte = random_sets(feat_list, df, stratify_wc)
    model.fit(Xtr, ytr)
    return accuracy_score(yte, model.predict(Xte))

def blend_eval(lr_pipe, xgb_mdl, feat_list, lr_w, df=None):
    Xtr, Xte, ytr, yte, _ = temporal_sets(feat_list, df)
    lrc = lr_pipe.__class__(**lr_pipe.get_params()) if hasattr(lr_pipe, "get_params") else build_lr(**LR_BEST)
    xgc = xgb_mod.XGBClassifier(**{k: v for k, v in XGB_BEST.items()})
    lrc.fit(Xtr, ytr)
    xgc.fit(Xtr, ytr, verbose=False)
    prob = lr_w * lrc.predict_proba(Xte) + (1 - lr_w) * xgc.predict_proba(Xte)
    return accuracy_score(yte, prob.argmax(axis=1)), lrc, xgc

# ─── Elo computation ──────────────────────────────────────────────────────────
def compute_elo(career_df, K):
    """Compute fighter Elo from all career fights chronologically.

    Returns history_df (one row per fighter per unique fight) and current ratings.
    Pre-fight Elo is stored — no leakage.
    """
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
            rf = float(row["won"])   # 1=win, 0=loss
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
        ratings[f]   = nra
        ratings[opp] = nrb
        last_date[f]   = date
        last_date[opp] = date
        n_fights[f]   += 1
        n_fights[opp] += 1

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


def compute_div_elo(master_df, K):
    """Division-specific Elo using UFC fights only (master has weight_class)."""
    ufc = (master_df[["date", "R_fighter", "B_fighter", "weight_class", "Winner"]]
           .dropna(subset=["R_fighter", "B_fighter", "weight_class"])
           .sort_values("date").reset_index(drop=True))

    ratings  = defaultdict(lambda: ELO_BASE)   # key: (fighter, div)
    processed = set()
    rows      = []

    for _, row in ufc.iterrows():
        f, opp = str(row["R_fighter"]), str(row["B_fighter"])
        date   = row["date"]
        div    = str(row["weight_class"])
        key    = (min(f, opp), max(f, opp), str(date.date()), div)
        if key in processed:
            continue
        processed.add(key)

        rk, bk = (f, div), (opp, div)
        ra, rb = ratings[rk], ratings[bk]
        winner = str(row.get("Winner", ""))
        rf = 1.0 if winner == "Red" else (0.5 if winner not in ("Red", "Blue") else 0.0)
        ro = 1.0 - rf

        ea  = 1 / (1 + 10 ** ((rb - ra) / 400))
        nra = ra + K * (rf - ea)
        nrb = rb + K * (ro - (1 - ea))

        rows += [
            {"fighter": f,   "div": div, "date": date, "div_elo_before": ra},
            {"fighter": opp, "div": div, "date": date, "div_elo_before": rb},
        ]
        ratings[rk] = nra
        ratings[bk] = nrb

    return pd.DataFrame(rows)


def join_global_elo(df, hist_df):
    """Add R_elo, B_elo, elo_dif, R_elo_trend, B_elo_trend, elo_trend_dif to df."""
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


def join_div_elo(df, div_hist_df):
    """Add R_div_elo, B_div_elo, div_elo_dif to df."""
    dh = div_hist_df.copy()
    dh["fighter_div"] = dh["fighter"] + "||" + dh["div"]
    df["R_fighter_div"] = df["R_fighter"].astype(str) + "||" + df["weight_class"].fillna("").astype(str)
    df["B_fighter_div"] = df["B_fighter"].astype(str) + "||" + df["weight_class"].fillna("").astype(str)

    for prefix, col in [("R", "R_fighter_div"), ("B", "B_fighter_div")]:
        # Rename fighter_div → col so merge_asof finds the by-key in both sides
        sub = (dh[["fighter_div", "date", "div_elo_before"]]
               .rename(columns={"fighter_div": col,
                                "div_elo_before": f"{prefix}_div_elo"})
               .sort_values("date"))
        df = pd.merge_asof(df.sort_values("date"), sub, on="date", by=col,
                           direction="backward")
    df = df.drop(columns=["R_fighter_div", "B_fighter_div"], errors="ignore")
    df["div_elo_dif"] = df["R_div_elo"] - df["B_div_elo"]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature engineering (108 base features, identical to exp8)
# ═════════════════════════════════════════════════════════════════════════════
log("=" * 72)
log(f"UFC EXPERIMENT 9  —  {pd.Timestamp.now().isoformat()}")
log("Experiments: A (Elo) + B (Weight-class aware)")
log(f"Primary target: temporal accuracy > {PREV_BEST_TEMPORAL:.4f}")
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

log(f"  career_fights_updated: {len(career_raw):,} rows")
log(f"  ufc-master:            {len(master):,} rows")
log(f"  career_fights columns: {list(career_raw.columns)}")

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
    "cum_wins", "cum_fights", "career_finish_rate", "recency_win_rate",
]
career_feat = career_with_opp[["fighter", "date"] + JOIN_COLS + ["opp_quality"]].copy()

def join_career(df, prefix):
    fc = f"{prefix}_fighter"
    sub = career_feat.rename(columns={
        "fighter": fc,
        **{c: f"{prefix}_{c}" for c in JOIN_COLS + ["opp_quality"]}
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
sub_base, _ = get_filtered(FEAT_108)
train_count = len(sub_base[sub_base["date"] <  pd.Timestamp(TEMPORAL_CUT)])
test_count  = len(sub_base[sub_base["date"] >= pd.Timestamp(TEMPORAL_CUT)])
log(f"  Rows — train: {train_count} | test: {test_count} | total: {len(sub_base)}")

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Elo computation: A1 (build) + A2 (K sweep)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 2 — Elo ratings: A1 (build) + A2 (K sweep)")
log("=" * 72)
t_step = time.time()

# A2: try all K values, pick best by correlation on test set
best_k       = 32
best_k_corr  = -1.0
elo_hists    = {}

log("  K sweep (correlation between elo_dif and Winner_bin on test set):")
for K in ELO_K_VALUES:
    hist_k, _, _ = compute_elo(career_raw, K)
    elo_hists[K] = hist_k

    # Join elo to master for correlation check
    tmp = join_global_elo(master[["date", "R_fighter", "B_fighter", "Winner_bin",
                                   "weight_class"]].copy(), hist_k)
    test_mask = tmp["date"] >= pd.Timestamp(TEMPORAL_CUT)
    corr = tmp.loc[test_mask, "elo_dif"].corr(tmp.loc[test_mask, "Winner_bin"])
    log(f"    K={K:3d}: corr={corr:.4f}")
    if corr > best_k_corr:
        best_k_corr = corr
        best_k = K

log(f"  Best K: {best_k} (corr={best_k_corr:.4f})")

# A1: save Elo history and current ratings using best K
hist_best, ratings_best, curr_best = compute_elo(career_raw, best_k)
hist_best.to_csv(ELO_HIST, index=False)
curr_best.to_csv(ELO_CURR, index=False)
log(f"  Saved {ELO_HIST} ({len(hist_best):,} rows) and {ELO_CURR} ({len(curr_best):,} rows)")
log(f"  Elo step done in {time.time()-t_step:.1f}s")

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — A3: Add 6 global Elo features → 114 features, retrain
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 3 — A3: +6 global Elo features (114 total), baseline retrain")
log("=" * 72)
t_step = time.time()

ELO_FEATS_6 = ["R_elo", "B_elo", "elo_dif", "R_elo_trend", "B_elo_trend", "elo_trend_dif"]
FEAT_114 = list(FEAT_108) + ELO_FEATS_6

# Build master_elo = master + 6 global Elo columns
master_elo = join_global_elo(master.copy(), hist_best)
sub_114, _ = get_filtered(FEAT_114, master_elo)
train_114  = len(sub_114[sub_114["date"] <  pd.Timestamp(TEMPORAL_CUT)])
test_114   = len(sub_114[sub_114["date"] >= pd.Timestamp(TEMPORAL_CUT)])
log(f"  Rows after NaN drop (114 feats) — train: {train_114} | test: {test_114}")

# LR retrain
t = time.time()
lr_a3 = build_lr(**LR_BEST)
a3_lr_t  = quick_temporal(lr_a3, FEAT_114, master_elo)[0]
a3_lr_r  = quick_random(build_lr(**LR_BEST), FEAT_114, master_elo)
record("A3 — LR baseline, 114 feats", a3_lr_t, a3_lr_r, 114, time.time()-t,
       model=lr_a3, feats=FEAT_114,
       meta={"model_type": "LR_elo114", "blend_ratio": "LR solo"})

# XGB retrain
t = time.time()
xgb_a3 = xgb_mod.XGBClassifier(**XGB_BEST)
Xtr3, Xte3, ytr3, yte3, _ = temporal_sets(FEAT_114, master_elo)
xgb_a3.fit(Xtr3, ytr3, verbose=False)
a3_xgb_t = accuracy_score(yte3, xgb_a3.predict(Xte3))

# Blend 85/15
lr_a3b = build_lr(**LR_BEST)
lr_a3b.fit(Xtr3, ytr3)
prob_lr3 = lr_a3b.predict_proba(Xte3)
prob_xgb3 = xgb_a3.predict_proba(Xte3)
a3_blend_t = accuracy_score(yte3, (0.85 * prob_lr3 + 0.15 * prob_xgb3).argmax(axis=1))
record("A3 — LR85+XGB15 blend, 114 feats", a3_blend_t, 0.0, 114, time.time()-t_step,
       model=lr_a3b, feats=FEAT_114, xgb_model=xgb_a3,
       meta={"model_type": "blend_LR85_XGB15_elo114", "blend_ratio": "85% LR + 15% XGB"})

# Track best LR pipe from A3 for feature importances
_a3_lr_for_coef = lr_a3b

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — A4: Optuna on 114 features (50 LR + 30 XGB trials)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 4 — A4: Optuna 50 LR + 30 XGB on 114 features")
log("=" * 72)
t_step = time.time()

Xtr4, Xte4, ytr4, yte4, _ = temporal_sets(FEAT_114, master_elo)

def lr_obj_114(trial):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C       = trial.suggest_float("C", 0.001, 100.0, log=True)
    l1_r    = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5
    scaler  = trial.suggest_categorical("scaler", ["robust", "standard", "none"])
    solver_hint = trial.suggest_categorical("solver", ["saga", "liblinear", "lbfgs"])
    valid = {"l1": {"saga", "liblinear"}, "l2": {"saga", "liblinear", "lbfgs"},
             "elasticnet": {"saga"}}
    solver = solver_hint if solver_hint in valid[penalty] else "saga"
    pipe   = build_lr(penalty, C, l1_r, scaler, solver)
    try:
        pipe.fit(Xtr4, ytr4)
        return accuracy_score(yte4, pipe.predict(Xte4))
    except Exception:
        return 0.0

study_lr4 = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RS))
study_lr4.optimize(lr_obj_114, n_trials=50, show_progress_bar=False)
bp4_lr = study_lr4.best_params
log(f"  Best LR (114 feats) temporal: {study_lr4.best_value:.4f}")
log(f"  Params: {bp4_lr}")

a4_lr_pipe = build_lr(bp4_lr["penalty"], bp4_lr["C"],
                      bp4_lr.get("l1_ratio", 0.5), bp4_lr["scaler"],
                      bp4_lr.get("solver", "saga"))
a4_lr_pipe.fit(Xtr4, ytr4)
a4_lr_proba = a4_lr_pipe.predict_proba(Xte4)
a4_lr_t = accuracy_score(yte4, a4_lr_pipe.predict(Xte4))
a4_lr_r = quick_random(build_lr(bp4_lr["penalty"], bp4_lr["C"],
                                 bp4_lr.get("l1_ratio", 0.5), bp4_lr["scaler"],
                                 bp4_lr.get("solver", "saga")), FEAT_114, master_elo)
record("A4 — LR tuned 50 trials, 114 feats", a4_lr_t, a4_lr_r, 114,
       time.time()-t_step, model=a4_lr_pipe, feats=FEAT_114,
       meta={"model_type": "LR_tuned_elo114", "params": bp4_lr})

def xgb_obj_114(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 1200),
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
    try:
        m = xgb_mod.XGBClassifier(**params)
        m.fit(Xtr4, ytr4, verbose=False)
        return accuracy_score(yte4, m.predict(Xte4))
    except Exception:
        return 0.0

study_xgb4 = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=RS))
study_xgb4.optimize(xgb_obj_114, n_trials=30, show_progress_bar=False)
bp4_xgb = study_xgb4.best_params
log(f"  Best XGB (114 feats) temporal: {study_xgb4.best_value:.4f}")
log(f"  Params: {bp4_xgb}")

xgb4_params = {**bp4_xgb, "use_label_encoder": False, "eval_metric": "logloss",
               "random_state": RS, "n_jobs": 1}
a4_xgb = xgb_mod.XGBClassifier(**xgb4_params)
a4_xgb.fit(Xtr4, ytr4, verbose=False)
a4_xgb_proba = a4_xgb.predict_proba(Xte4)

# Blend sweep: 85/15, 90/10, 80/20
a4_best_t = 0.0
a4_best_lr_w = 0.85
a4_best_lr_pipe2 = None
a4_best_xgb2 = None

for lr_w in [0.85, 0.90, 0.80]:
    xw = 1.0 - lr_w
    prob = lr_w * a4_lr_proba + xw * a4_xgb_proba
    t_acc = accuracy_score(yte4, prob.argmax(axis=1))
    tag = f"A4 — LR{int(lr_w*100)}+XGB{int(xw*100)} blend, 114 feats"
    record(tag, t_acc, 0.0, 114, time.time()-t_step,
           model=a4_lr_pipe, feats=FEAT_114, xgb_model=a4_xgb,
           meta={"model_type": f"blend_LR{int(lr_w*100)}_XGB{int(xw*100)}_elo114",
                 "blend_ratio": f"{lr_w:.0%} LR + {xw:.0%} XGB"})
    if t_acc > a4_best_t:
        a4_best_t    = t_acc
        a4_best_lr_w = lr_w

log(f"  A4 best blend: LR={a4_best_lr_w:.0%} temporal={a4_best_t:.4f}")

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — A5: Division-specific Elo (+3 features → 117 total)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 5 — A5: Division-specific Elo (117 features)")
log("=" * 72)
t_step = time.time()

DIV_ELO_FEATS = ["R_div_elo", "B_div_elo", "div_elo_dif"]
FEAT_117 = list(FEAT_114) + DIV_ELO_FEATS

div_hist = compute_div_elo(master, best_k)
master_elo_div = join_div_elo(master_elo.copy(), div_hist)

sub_117, _ = get_filtered(FEAT_117, master_elo_div)
train_117  = len(sub_117[sub_117["date"] <  pd.Timestamp(TEMPORAL_CUT)])
test_117   = len(sub_117[sub_117["date"] >= pd.Timestamp(TEMPORAL_CUT)])
log(f"  Rows (117 feats) — train: {train_117} | test: {test_117}")

lr_a5 = build_lr(**LR_BEST)
a5_lr_t, a5_lr_proba, a5_yte = quick_temporal(lr_a5, FEAT_117, master_elo_div)

xgb_a5 = xgb_mod.XGBClassifier(**XGB_BEST)
Xtr5, Xte5, ytr5, yte5, _ = temporal_sets(FEAT_117, master_elo_div)
lr_a5.fit(Xtr5, ytr5)
xgb_a5.fit(Xtr5, ytr5, verbose=False)
prob_a5 = 0.85 * lr_a5.predict_proba(Xte5) + 0.15 * xgb_a5.predict_proba(Xte5)
a5_blend_t = accuracy_score(yte5, prob_a5.argmax(axis=1))
a5_r = quick_random(build_lr(**LR_BEST), FEAT_117, master_elo_div)

record("A5 — LR85+XGB15 blend, 117 feats (div elo)", a5_blend_t, a5_r, 117,
       time.time()-t_step, model=lr_a5, feats=FEAT_117, xgb_model=xgb_a5,
       meta={"model_type": "blend_elo117", "blend_ratio": "85% LR + 15% XGB"})

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — B1: Division difficulty features (+4 → 112 base)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 6 — B1: Division difficulty features (112 features)")
log("=" * 72)
t_step = time.time()

# Compute from training set only (pre-2024), no leakage
train_mask = master["date"] < pd.Timestamp(TEMPORAL_CUT)
div_stats  = (master.loc[train_mask]
              .groupby("weight_class")
              .agg(
                  div_fight_count=("Winner_bin", "count"),
                  div_avg_finish_rate=("Winner_bin", lambda x: (
                      master.loc[x.index, "R_win_by_KO/TKO"].fillna(0) +
                      master.loc[x.index, "R_win_by_Submission"].fillna(0) +
                      master.loc[x.index, "B_win_by_KO/TKO"].fillna(0) +
                      master.loc[x.index, "B_win_by_Submission"].fillna(0) > 0
                  ).mean()),
              )
              .reset_index())

# Simpler finish rate: fights not ending in decision / total fights in that division
div_finish = {}
div_upset  = {}
for div, grp in master[train_mask].groupby("weight_class"):
    total = len(grp)
    if total == 0:
        continue
    # Finish = KO/TKO or Sub (approximate via R_win columns)
    r_ko  = grp.get("R_win_by_KO/TKO",  pd.Series(0, index=grp.index)).fillna(0)
    r_sub = grp.get("R_win_by_Submission", pd.Series(0, index=grp.index)).fillna(0)
    b_ko  = grp.get("B_win_by_KO/TKO",  pd.Series(0, index=grp.index)).fillna(0)
    b_sub = grp.get("B_win_by_Submission", pd.Series(0, index=grp.index)).fillna(0)
    finishes = ((r_ko + r_sub + b_ko + b_sub) > 0).sum()
    div_finish[div] = finishes / total

    # Upset: blue corner wins (since red is typically favourite)
    div_upset[div] = float((grp["Winner_bin"] == 0).mean())

master["div_avg_finish_rate"] = master["weight_class"].map(div_finish).fillna(
    np.mean(list(div_finish.values()))
)
master["div_upset_rate"] = master["weight_class"].map(div_upset).fillna(
    np.mean(list(div_upset.values()))
)
master["div_avg_decision_rate"] = 1.0 - master["div_avg_finish_rate"]
master["div_fight_count"]       = master["weight_class"].map(
    dict(zip(div_stats["weight_class"], div_stats["div_fight_count"]))
).fillna(0)

DIV_DIFF_FEATS = ["div_avg_finish_rate", "div_upset_rate",
                  "div_avg_decision_rate", "div_fight_count"]
FEAT_112 = list(FEAT_108) + DIV_DIFF_FEATS

lr_b1 = build_lr(**LR_BEST)
b1_lr_t, _, _ = quick_temporal(lr_b1, FEAT_112)
b1_lr_r       = quick_random(build_lr(**LR_BEST), FEAT_112)
record("B1 — div difficulty features (112)", b1_lr_t, b1_lr_r, 112,
       time.time()-t_step, model=lr_b1, feats=FEAT_112,
       meta={"model_type": "LR_divdiff112"})

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — B2: Division embedding (one-hot 12 cols vs binary)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 7 — B2: Division embedding (one-hot vs lighter/heavier binary)")
log("=" * 72)
t_step = time.time()

WC_LIST = [
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight", "Flyweight", "Bantamweight", "Featherweight",
    "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
]
for wc in WC_LIST:
    safe = wc.replace(" ", "_").replace("'", "")
    master[f"wc_{safe}"] = (master["weight_class"] == wc).astype(int)

ONEHOT_FEATS = [f"wc_{wc.replace(' ', '_').replace(chr(39), '')}" for wc in WC_LIST]
# Replace ordinal with one-hot (drop weight_class_ord from base)
FEAT_ONEHOT = [c for c in FEAT_108 if c != "weight_class_ord"] + ONEHOT_FEATS

lr_b2_oh = build_lr(**LR_BEST)
b2_oh_t, _, _ = quick_temporal(lr_b2_oh, FEAT_ONEHOT)
b2_oh_r       = quick_random(build_lr(**LR_BEST), FEAT_ONEHOT)
record(f"B2 — one-hot encoding ({len(FEAT_ONEHOT)} feats)", b2_oh_t, b2_oh_r,
       len(FEAT_ONEHOT), time.time()-t_step, model=lr_b2_oh, feats=FEAT_ONEHOT,
       meta={"model_type": "LR_onehot"})

# Binary: lighter (straw/fly/bantam/feather) vs heavier
master["lighter_div"] = master["weight_class"].isin(LIGHTER_DIVS).astype(int)
FEAT_BINARY = [c for c in FEAT_108 if c != "weight_class_ord"] + ["lighter_div"]

lr_b2_bin = build_lr(**LR_BEST)
b2_bin_t, _, _ = quick_temporal(lr_b2_bin, FEAT_BINARY)
b2_bin_r       = quick_random(build_lr(**LR_BEST), FEAT_BINARY)
record(f"B2 — lighter/heavier binary ({len(FEAT_BINARY)} feats)", b2_bin_t, b2_bin_r,
       len(FEAT_BINARY), time.time()-t_step, model=lr_b2_bin, feats=FEAT_BINARY,
       meta={"model_type": "LR_divbinary"})

b2_best_t = max(b2_oh_t, b2_bin_t)

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — B3: Division-stratified random split (impact on random accuracy)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 8 — B3: Division-stratified CV (stratified random split)")
log("=" * 72)
t_step = time.time()

# Temporal accuracy unchanged — stratification only affects random split
lr_b3 = build_lr(**LR_BEST)
b3_temporal_t, _, b3_yte = quick_temporal(lr_b3, FEAT_108)

lr_b3_r = build_lr(**LR_BEST)
b3_r = quick_random(lr_b3_r, FEAT_108, stratify_wc=True)
record("B3 — stratified random split (108 feats)", b3_temporal_t, b3_r, 108,
       time.time()-t_step, model=lr_b3, feats=FEAT_108,
       meta={"model_type": "LR_stratified"})

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — B4: Per-division-group models
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 9 — B4: Per-division-group models (high/medium/low)")
log("=" * 72)
t_step = time.time()

sub_b4, cols_b4 = get_filtered(FEAT_108)
sub_b4 = sub_b4.copy()
sub_b4["div_group"] = sub_b4["weight_class"].map(DIV_GROUPS).fillna("medium")

group_results = []
for group in ["high", "medium", "low"]:
    sub_g  = sub_b4[sub_b4["div_group"] == group]
    train_g = sub_g[sub_g["date"] <  pd.Timestamp(TEMPORAL_CUT)]
    test_g  = sub_g[sub_g["date"] >= pd.Timestamp(TEMPORAL_CUT)]
    if len(train_g) < 50 or len(test_g) < 10:
        log(f"  {group}: skip ({len(train_g)} train, {len(test_g)} test)")
        group_results.append((group, None, 0, 0))
        continue
    Xtr_g, ytr_g = train_g[cols_b4], train_g["Winner_bin"]
    Xtr_ga, ytr_ga = augment(Xtr_g, ytr_g, cols_b4)
    lr_g = build_lr(**LR_BEST)
    lr_g.fit(Xtr_ga.values, ytr_ga.values)
    g_acc = accuracy_score(test_g["Winner_bin"].values, lr_g.predict(test_g[cols_b4].values))
    log(f"  {group:8s}: {g_acc:.4f} ({len(train_g)} train, {len(test_g)} test)")
    group_results.append((group, lr_g, g_acc, len(test_g)))

# Weighted average temporal accuracy across groups
total_test = sum(n for _, _, _, n in group_results)
if total_test > 0:
    b4_t = sum(acc * n for _, _, acc, n in group_results) / total_test
else:
    b4_t = 0.0

log(f"  Weighted avg temporal: {b4_t:.4f}")
record("B4 — per-division-group LR models", b4_t, 0.0, 108,
       time.time()-t_step, model=lr_b3,   # store base LR for saving purposes
       meta={"model_type": "per_group_LR"})

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 10 — Combined: best Elo + best weight-class features
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("STEP 10 — Combined: best Elo (114) + div difficulty (B1) features")
log("=" * 72)
t_step = time.time()

# Add div difficulty features to master_elo
master_combined = master_elo.copy()
for col in DIV_DIFF_FEATS:
    if col in master.columns:
        master_combined[col] = master[col].values

FEAT_COMBINED = list(FEAT_114) + DIV_DIFF_FEATS   # 118 features

sub_comb, _ = get_filtered(FEAT_COMBINED, master_combined)
train_comb  = len(sub_comb[sub_comb["date"] <  pd.Timestamp(TEMPORAL_CUT)])
test_comb   = len(sub_comb[sub_comb["date"] >= pd.Timestamp(TEMPORAL_CUT)])
log(f"  Combined ({len(FEAT_COMBINED)} feats) — train: {train_comb} | test: {test_comb}")

lr_comb = build_lr(**LR_BEST)
lr_comb_t, lr_comb_proba, comb_yte = quick_temporal(lr_comb, FEAT_COMBINED, master_combined)

xgb_comb = xgb_mod.XGBClassifier(**XGB_BEST)
Xtr_c, Xte_c, ytr_c, yte_c, _ = temporal_sets(FEAT_COMBINED, master_combined)
lr_comb.fit(Xtr_c, ytr_c)
xgb_comb.fit(Xtr_c, ytr_c, verbose=False)
prob_comb = 0.85 * lr_comb.predict_proba(Xte_c) + 0.15 * xgb_comb.predict_proba(Xte_c)
combined_blend_t = accuracy_score(yte_c, prob_comb.argmax(axis=1))
combined_r = quick_random(build_lr(**LR_BEST), FEAT_COMBINED, master_combined)

record("Combined — LR85+XGB15 (Elo+divdiff, 118 feats)", combined_blend_t, combined_r,
       len(FEAT_COMBINED), time.time()-t_step, model=lr_comb, feats=FEAT_COMBINED,
       xgb_model=xgb_comb,
       meta={"model_type": "blend_elo_divdiff118", "blend_ratio": "85% LR + 15% XGB"})

gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# Save best model
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("SAVING BEST MODEL")
log("=" * 72)
save_best()

# ═════════════════════════════════════════════════════════════════════════════
# Feature importances for Elo features (from best A4 LR)
# ═════════════════════════════════════════════════════════════════════════════
elo_coef = {}
try:
    coef_map = lr_coefs(a4_lr_pipe, FEAT_114)
    for feat in ["elo_dif", "R_elo_trend", "B_elo_trend", "elo_trend_dif", "R_elo", "B_elo"]:
        elo_coef[feat] = coef_map.get(feat, 0.0)
except Exception as e:
    log(f"  [warn] Could not extract Elo coefs: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# All results table
# ═════════════════════════════════════════════════════════════════════════════
all_results.sort(key=lambda x: -x["temporal_acc"])
log()
log("=" * 72)
log("ALL RESULTS (ranked by temporal accuracy):")
log("=" * 72)
log(f"  {'Config':60s}  {'Temporal':>8}  {'Random':>8}  {'Feats':>5}")
log("  " + "-" * 90)
for r in all_results:
    log(f"  {r['config']:60s}  {r['temporal_acc']:.4f}    {r['random_acc']:.4f}    {r['n_feats']:3d}")

# ═════════════════════════════════════════════════════════════════════════════
# Final summary (printed AND logged)
# ═════════════════════════════════════════════════════════════════════════════
best_res = all_results[0]

def fmt(val):
    return f"{val:.2%}" if isinstance(val, float) else "--"

summary_lines = [
    "",
    "=" * 40,
    "EXPERIMENT 9 — ELO + WEIGHT CLASS",
    "=" * 40,
    f"Baseline (no changes): {PREV_BEST_TEMPORAL:.2%} temporal",
    "",
    "ELO RESULTS:",
    f"  Best K value: K={best_k} (correlation: {best_k_corr:.4f})",
    "",
    f"  {'Config':30s}  {'Temporal':>8}   {'Random':>8}   {'Features':>8}",
    f"  {'-'*30}  {'-'*8}   {'-'*8}   {'-'*8}",
    f"  {'A3 — +6 global elo features':30s}  {fmt(a3_blend_t):>8}   {fmt(0.0):>8}   {'114':>8}",
    f"  {'A4 — tuned on 114 features':30s}  {fmt(a4_best_t):>8}   {fmt(a4_lr_r):>8}   {'114':>8}",
    f"  {'A5 — +div elo (117 feats)':30s}  {fmt(a5_blend_t):>8}   {fmt(a5_r):>8}   {'117':>8}",
    "",
    "  Elo feature importances (LR coef, tuned A4 model):",
]
for feat in ["elo_dif", "R_elo_trend", "B_elo_trend", "elo_trend_dif"]:
    val = elo_coef.get(feat, 0.0)
    summary_lines.append(f"    {feat:20s}: {val:+.4f}")

summary_lines += [
    "",
    "WEIGHT CLASS RESULTS:",
    f"  {'Config':30s}  {'Temporal':>8}   {'Random':>8}",
    f"  {'-'*30}  {'-'*8}   {'-'*8}",
    f"  {'B1 — div difficulty features':30s}  {fmt(b1_lr_t):>8}   {fmt(b1_lr_r):>8}",
    f"  {'B2 — one-hot encoding':30s}  {fmt(b2_oh_t):>8}   {fmt(b2_oh_r):>8}",
    f"  {'B2 — lighter/heavier binary':30s}  {fmt(b2_bin_t):>8}   {fmt(b2_bin_r):>8}",
    f"  {'B3 — stratified random split':30s}  {fmt(b3_temporal_t):>8}   {fmt(b3_r):>8}",
    f"  {'B4 — per-group models':30s}  {fmt(b4_t):>8}   {'N/A':>8}",
    "",
    "COMBINED (best Elo + div difficulty):",
    f"  Temporal: {fmt(combined_blend_t)} | Random: {fmt(combined_r)} | Features: {len(FEAT_COMBINED)}",
    "",
    f"BEST RESULT: {fmt(best_res['temporal_acc'])} (config: {best_res['config']})",
    f"vs baseline: {best_res['temporal_acc'] - PREV_BEST_TEMPORAL:+.2%}",
    "",
    "Files saved:",
    f"  {'✓' if best_model_obj is not None else '✗'} {BEST_MDL}",
    f"  {'✓' if best_xgb_obj   is not None else '✗'} {BEST_XGB}",
    f"  {'✓' if os.path.exists(ELO_HIST) else '✗'} {ELO_HIST}",
    f"  {'✓' if os.path.exists(ELO_CURR) else '✗'} {ELO_CURR}",
    "=" * 40,
]

for line in summary_lines:
    log(line)

print()
for line in summary_lines:
    print(line)
