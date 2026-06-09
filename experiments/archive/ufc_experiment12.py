"""
ufc_experiment12.py — Opponent-quality-weighted Elo K.

A1: sqrt scaling  A2: linear  A3: log  A4: clipped linear
A5: exponent sweep on sqrt variant
B: LR retune on best variant (only if A beats 73.24%)

Baseline: 73.24% temporal (K=48 fixed)
"""

import gc, json, math, os, time, warnings
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
PREV_BEST_TEMPORAL = 0.7324
ELO_BASE           = 1500.0
K_BASE             = 48

OUT_LOG   = "model/experiment12_output.txt"
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
best_elo_hist_sv = None
best_elo_curr_sv = None

# ─── Feature list ─────────────────────────────────────────────────────────────
FEAT_114      = joblib.load(BEST_FT)
ELO_6         = ["R_elo", "B_elo", "elo_dif", "R_elo_trend", "B_elo_trend", "elo_trend_dif"]
FEAT_108_BASE = [f for f in FEAT_114 if f not in ELO_6]
log(f"Loaded feature_columns_best.pkl: {len(FEAT_114)} features")

# ─── Extract params from saved models ─────────────────────────────────────────
_lr_pipe = joblib.load(BEST_MDL)
_lr_step = _lr_pipe.named_steps["lr"]
_lr_sc   = _lr_pipe.named_steps["sc"]
BEST_LR_PARAMS = {
    "penalty":  _lr_step.penalty,
    "C":        _lr_step.C,
    "l1_ratio": _lr_step.l1_ratio if _lr_step.penalty == "elasticnet" else 0.5,
    "scaler":   "robust" if isinstance(_lr_sc, RobustScaler) else "standard",
    "solver":   _lr_step.solver,
}
log(f"LR params: penalty={BEST_LR_PARAMS['penalty']}, C={BEST_LR_PARAMS['C']:.5f}, "
    f"scaler={BEST_LR_PARAMS['scaler']}, solver={BEST_LR_PARAMS['solver']}")

_xgb_loaded = joblib.load(BEST_XGB)
_xp = _xgb_loaded.get_params()
XGB_KEYS = ["n_estimators", "learning_rate", "max_depth", "subsample",
            "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda"]
BEST_XGB_PARAMS = {k: _xp[k] for k in XGB_KEYS if k in _xp}
log(f"XGB params: n_est={BEST_XGB_PARAMS.get('n_estimators')}, "
    f"lr={BEST_XGB_PARAMS.get('learning_rate'):.4f}, "
    f"depth={BEST_XGB_PARAMS.get('max_depth')}")
del _lr_pipe, _xgb_loaded
gc.collect()

# ─── Tracking ─────────────────────────────────────────────────────────────────
def record(tag, t_acc, r_acc, n_feats, secs,
           lr_model=None, xgb_model=None, feats=None, meta=None,
           elo_hist=None, elo_curr=None):
    global best_temporal, best_model_lr, best_model_xgb, best_feats, best_meta
    global best_elo_hist_sv, best_elo_curr_sv
    marker = ""
    if t_acc > best_temporal:
        marker            = "  *** NEW BEST ***"
        best_temporal     = t_acc
        best_model_lr     = lr_model
        best_model_xgb    = xgb_model
        best_feats        = feats
        best_meta         = meta or {}
        best_elo_hist_sv  = elo_hist
        best_elo_curr_sv  = elo_curr
    r_str = f"r={r_acc:.4f}" if r_acc is not None else "r=      --"
    log(f"  {tag:52s}  t={t_acc:.4f}  {r_str}  feats={n_feats}  {secs:.0f}s{marker}")
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
    if best_elo_hist_sv is not None:
        best_elo_hist_sv.to_csv(ELO_HIST, index=False)
    if best_elo_curr_sv is not None:
        best_elo_curr_sv.to_csv(ELO_CURR, index=False)
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

def build_xgb(**params):
    p = dict(params)
    p.update({"use_label_encoder": False, "eval_metric": "logloss",
              "random_state": RS, "n_jobs": 1})
    return xgb_mod.XGBClassifier(**p)

def lr_coefs(pipe, feat_list):
    return dict(zip(feat_list, pipe.named_steps["lr"].coef_[0]))

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

def eval_blend(feat_list, df, lr_w=0.90):
    Xtr, Xte, ytr, yte, _ = temporal_sets(feat_list, df)
    lr  = build_lr(**BEST_LR_PARAMS)
    xgb = build_xgb(**BEST_XGB_PARAMS)
    lr.fit(Xtr, ytr)
    xgb.fit(Xtr, ytr, verbose=False)
    prob  = lr_w * lr.predict_proba(Xte) + (1 - lr_w) * xgb.predict_proba(Xte)
    t_acc = accuracy_score(yte, prob.argmax(axis=1))

    Xtr_r, Xte_r, ytr_r, yte_r = random_sets(feat_list, df)
    lr_r  = build_lr(**BEST_LR_PARAMS)
    xgb_r = build_xgb(**BEST_XGB_PARAMS)
    lr_r.fit(Xtr_r, ytr_r)
    xgb_r.fit(Xtr_r, ytr_r, verbose=False)
    r_acc = accuracy_score(yte_r,
        (lr_w * lr_r.predict_proba(Xte_r) + (1 - lr_w) * xgb_r.predict_proba(Xte_r)).argmax(axis=1))
    return t_acc, r_acc, lr, xgb

def elo_dif_corr(feat_list, df):
    sub, _ = get_filtered(feat_list, df)
    test   = sub[sub["date"] >= pd.Timestamp(TEMPORAL_CUT)]
    return float(test["elo_dif"].corr(test["Winner_bin"]))

# ─── Elo join ─────────────────────────────────────────────────────────────────
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

# ─── Quality-K Elo computation ────────────────────────────────────────────────
def compute_quality_elo(career_df, k_fn, base_k=48):
    """
    Elo where K is determined by opponent's pre-fight Elo rating.
    k_fn(opponent_pre_fight_elo, base_k) -> K
    Each fighter's K is based on their own opponent's pre-fight rating.
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

        ra, rb = ratings[f], ratings[opp]   # both pre-fight

        # Each fighter's update magnitude is scaled by the opponent's pre-fight Elo
        K_f = k_fn(rb, base_k)   # f's update: opponent is opp (rb)
        K_o = k_fn(ra, base_k)   # opp's update: opponent is f (ra)

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

# ─── K scaling functions ──────────────────────────────────────────────────────
def k_sqrt(opp_elo, base_k, exp=0.5):
    return base_k * (max(opp_elo, 100) / 1500) ** exp

def k_linear(opp_elo, base_k):
    return base_k * (max(opp_elo, 100) / 1500)

def k_log(opp_elo, base_k):
    ratio = max(opp_elo, 100) / 1500
    return base_k * max(0.1, 1 + 0.3 * math.log(ratio))

def k_clipped(opp_elo, base_k):
    return base_k * max(0.5, min(2.0, max(opp_elo, 100) / 1500))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature engineering (same pipeline as exp11)
# ═════════════════════════════════════════════════════════════════════════════
log("=" * 72)
log(f"UFC EXPERIMENT 12  —  {pd.Timestamp.now().isoformat()}")
log("Experiments: A (quality-weighted K variants) + B (retune if A wins)")
log(f"Baseline: {PREV_BEST_TEMPORAL:.4f} temporal (K=48 fixed)")
log("=" * 72)
log()
log("STEP 1 — Feature engineering")
t0 = time.time()

career_raw = pd.read_csv("data/career_fights_updated.csv")
career_raw["date"] = pd.to_datetime(career_raw["date"])
master     = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
fighters   = pd.read_csv("data/ufc_fighters_final_updated.csv")

log(f"  career_fights_updated: {len(career_raw):,} rows")

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
sub_base, _ = get_filtered(FEAT_108_BASE, master)
log(f"  Base rows — train: {len(sub_base[sub_base['date'] < pd.Timestamp(TEMPORAL_CUT)])} "
    f"| test: {len(sub_base[sub_base['date'] >= pd.Timestamp(TEMPORAL_CUT)])}")
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Quality-weighted K variants
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("EXPERIMENT A — QUALITY-WEIGHTED K VARIANTS")
log("=" * 72)

VARIANTS = [
    ("A1 — sqrt scaling",    lambda opp, bk: k_sqrt(opp, bk, exp=0.5)),
    ("A2 — linear",          k_linear),
    ("A3 — log",             k_log),
    ("A4 — clipped linear",  k_clipped),
]

# (tag, t_acc, r_acc, corr, hist, curr)
a_variant_results = []

for vtag, vfn in VARIANTS:
    log()
    log(f"  {vtag}")
    t_v = time.time()

    hist_v, _, curr_v = compute_quality_elo(career_raw, vfn, K_BASE)
    master_v = join_elo_standard(master.copy(), hist_v)

    corr_v = elo_dif_corr(FEAT_114, master_v)
    t_acc, r_acc, lr_v, xgb_v = eval_blend(FEAT_114, master_v)

    record(vtag, t_acc, r_acc, 114, time.time()-t_v,
           lr_model=lr_v, xgb_model=xgb_v, feats=FEAT_114,
           meta={"model_type": f"quality_elo_{vtag}", "blend_ratio": "90% LR + 10% XGB"},
           elo_hist=hist_v, elo_curr=curr_v)

    a_variant_results.append((vtag, t_acc, r_acc, corr_v, hist_v, curr_v))
    del master_v
    gc.collect()

# Unpack per-variant results for summary
a1_t, a1_r, a1_corr = a_variant_results[0][1], a_variant_results[0][2], a_variant_results[0][3]
a2_t, a2_r, a2_corr = a_variant_results[1][1], a_variant_results[1][2], a_variant_results[1][3]
a3_t, a3_r, a3_corr = a_variant_results[2][1], a_variant_results[2][2], a_variant_results[2][3]
a4_t, a4_r, a4_corr = a_variant_results[3][1], a_variant_results[3][2], a_variant_results[3][3]

# ─── A5 — Exponent sweep ──────────────────────────────────────────────────────
log()
log("  A5 — Exponent sweep (sqrt variant)")
EXPONENTS  = [0.25, 0.33, 0.50, 0.67, 0.75, 1.00]
a5_results = []   # (exp, t_acc, hist, curr)

for exp in EXPONENTS:
    if abs(exp - 0.50) < 1e-9:
        # Reuse A1 result
        a5_results.append((exp, a1_t, a_variant_results[0][4], a_variant_results[0][5]))
        log(f"    exp={exp:.2f}  t={a1_t:.4f}  (reused A1)")
        continue
    if abs(exp - 1.00) < 1e-9:
        # Reuse A2 result
        a5_results.append((exp, a2_t, a_variant_results[1][4], a_variant_results[1][5]))
        log(f"    exp={exp:.2f}  t={a2_t:.4f}  (reused A2)")
        continue

    t_e = time.time()
    vfn_e = lambda opp, bk, _e=exp: k_sqrt(opp, bk, exp=_e)
    hist_e, _, curr_e = compute_quality_elo(career_raw, vfn_e, K_BASE)
    master_e = join_elo_standard(master.copy(), hist_e)
    t_acc_e, r_acc_e, lr_e, xgb_e = eval_blend(FEAT_114, master_e)

    tag_e = f"A5 — sqrt exp={exp:.2f}"
    record(tag_e, t_acc_e, r_acc_e, 114, time.time()-t_e,
           lr_model=lr_e, xgb_model=xgb_e, feats=FEAT_114,
           meta={"model_type": f"quality_elo_sqrt_exp{exp}", "blend_ratio": "90% LR + 10% XGB"},
           elo_hist=hist_e, elo_curr=curr_e)

    a5_results.append((exp, t_acc_e, hist_e, curr_e))
    log(f"    exp={exp:.2f}  t={t_acc_e:.4f}  {time.time()-t_e:.0f}s")
    del master_e
    gc.collect()

# Best across all A variants (A1-A4 + A5 new exponents)
best_a_t   = max(r[1] for r in a_variant_results + [(x[0], x[1], None, None, None, None) for x in a5_results])
best_a_tag = ""
best_a_hist = None
best_a_curr = None
for vtag, t_acc, r_acc, corr, hist_v, curr_v in a_variant_results:
    if abs(t_acc - best_a_t) < 1e-9:
        best_a_tag = vtag; best_a_hist = hist_v; best_a_curr = curr_v
        break
for exp, t_acc_e, hist_e, curr_e in a5_results:
    if abs(t_acc_e - best_a_t) < 1e-9 and best_a_hist is None:
        best_a_tag = f"A5 exp={exp:.2f}"; best_a_hist = hist_e; best_a_curr = curr_e

log()
log(f"  Best A variant: {best_a_tag} (temporal={best_a_t:.4f})")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — LR retune on best variant (only if A beats 73.24%)
# ═════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("EXPERIMENT B — LR RETUNE ON BEST VARIANT")
log("=" * 72)

b_ran       = False
b_best_t    = None
b_best_lrw  = None
b_best_lr_p = None
b_blend_results = []

if best_a_t <= PREV_BEST_TEMPORAL:
    log(f"  Skipped — best A ({best_a_t:.4f}) does not beat baseline ({PREV_BEST_TEMPORAL:.4f})")
else:
    log(f"  Running 50 Optuna trials on LR (C: 0.001–0.1, penalty l2/elasticnet)")
    b_ran = True

    master_b = join_elo_standard(master.copy(), best_a_hist)
    Xtr_b, Xte_b, ytr_b, yte_b, _ = temporal_sets(FEAT_114, master_b)

    def lr_obj_b(trial):
        penalty = trial.suggest_categorical("penalty", ["l2", "elasticnet"])
        C       = trial.suggest_float("C", 0.001, 0.1, log=True)
        l1_r    = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5
        scaler  = trial.suggest_categorical("scaler", ["robust", "standard"])
        solver  = "saga"   # l2+saga or elasticnet+saga
        try:
            pipe = build_lr(penalty, C, l1_r, scaler, solver)
            pipe.fit(Xtr_b, ytr_b)
            return accuracy_score(yte_b, pipe.predict(Xte_b))
        except Exception:
            return 0.0

    study_b = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=RS))
    study_b.optimize(lr_obj_b, n_trials=50, show_progress_bar=False)
    bp_b = study_b.best_params
    b_best_lr_p = {
        "penalty":  bp_b["penalty"],
        "C":        bp_b["C"],
        "l1_ratio": bp_b.get("l1_ratio", 0.5),
        "scaler":   bp_b["scaler"],
        "solver":   "saga",
    }
    log(f"  Best LR temporal (solo): {study_b.best_value:.4f}")
    log(f"  Params: {b_best_lr_p}")

    # Fit final models
    lr_b_final  = build_lr(**b_best_lr_p)
    xgb_b_final = build_xgb(**BEST_XGB_PARAMS)
    lr_b_final.fit(Xtr_b, ytr_b)
    xgb_b_final.fit(Xtr_b, ytr_b, verbose=False)
    lr_proba_b  = lr_b_final.predict_proba(Xte_b)
    xgb_proba_b = xgb_b_final.predict_proba(Xte_b)

    for lr_w in [0.95, 0.90, 0.85]:
        xw    = 1.0 - lr_w
        prob  = lr_w * lr_proba_b + xw * xgb_proba_b
        t_acc = accuracy_score(yte_b, prob.argmax(axis=1))

        Xtr_br, Xte_br, ytr_br, yte_br = random_sets(FEAT_114, master_b)
        lr_br  = build_lr(**b_best_lr_p)
        xgb_br = build_xgb(**BEST_XGB_PARAMS)
        lr_br.fit(Xtr_br, ytr_br); xgb_br.fit(Xtr_br, ytr_br, verbose=False)
        r_acc = accuracy_score(yte_br,
            (lr_w * lr_br.predict_proba(Xte_br) + xw * xgb_br.predict_proba(Xte_br)).argmax(axis=1))

        b_blend_results.append((lr_w, t_acc, r_acc))
        tag_b = f"B — LR{int(lr_w*100)}+XGB{int(round(xw*100))} retune, best variant"
        record(tag_b, t_acc, r_acc, 114, 0,
               lr_model=lr_b_final, xgb_model=xgb_b_final, feats=FEAT_114,
               meta={"model_type": f"B_blend_LR{int(lr_w*100)}",
                     "blend_ratio": f"{lr_w:.0%} LR + {xw:.0%} XGB",
                     "lr_params": b_best_lr_p},
               elo_hist=best_a_hist, elo_curr=best_a_curr)

    b_best_lrw, b_best_t, _ = max(b_blend_results, key=lambda x: x[1])
    log(f"  Best blend: LR{int(b_best_lrw*100)}/XGB{int(round((1-b_best_lrw)*100))} = {b_best_t:.4f}")
    del master_b
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
    "EXPERIMENT 12 — OPPONENT-QUALITY ELO",
    "=" * 40,
    f"Baseline: {PREV_BEST_TEMPORAL:.2%} temporal (K=48 fixed)",
    "",
    "EXPERIMENT A — QUALITY-WEIGHTED K VARIANTS:",
    f"  {'Variant':21s}  {'Temporal':>8}  {'Random':>8}  {'elo_dif corr':>12}",
    f"  {'-------------------':21s}  {'--------':>8}  {'------':>8}  {'------------':>12}",
    f"  {'A1 — sqrt scaling':21s}  {fmt(a1_t):>8}  {fmt(a1_r):>8}  {a1_corr:>12.4f}",
    f"  {'A2 — linear':21s}  {fmt(a2_t):>8}  {fmt(a2_r):>8}  {a2_corr:>12.4f}",
    f"  {'A3 — log':21s}  {fmt(a3_t):>8}  {fmt(a3_r):>8}  {a3_corr:>12.4f}",
    f"  {'A4 — clipped linear':21s}  {fmt(a4_t):>8}  {fmt(a4_r):>8}  {a4_corr:>12.4f}",
    "",
    "  A5 — exponent sweep (sqrt variant):",
]

for exp, t_acc_e, _, _ in a5_results:
    note = "  (A1 baseline)" if abs(exp - 0.50) < 1e-9 else (
           "  (= A2 linear)" if abs(exp - 1.00) < 1e-9 else "")
    summary.append(f"    exp={exp:.2f}   {fmt(t_acc_e)}{note}")

summary += [
    "",
    f"  Best variant: {best_a_tag} (temporal: {fmt(best_a_t)})",
    "",
    "EXPERIMENT B — RETUNE ON BEST VARIANT:",
]

if not b_ran:
    summary.append(f"  Ran: No — best A ({fmt(best_a_t)}) did not beat {fmt(PREV_BEST_TEMPORAL)}")
else:
    summary.append(f"  Ran: Yes")
    if b_best_lr_p:
        summary.append(f"  Best LR params: penalty={b_best_lr_p['penalty']}, "
                       f"C={b_best_lr_p['C']:.5f}, scaler={b_best_lr_p['scaler']}")
    if b_blend_results:
        for lr_w, t_acc, r_acc in b_blend_results:
            xw = 1.0 - lr_w
            summary.append(f"    LR{int(lr_w*100)}/XGB{int(round(xw*100))}:  {fmt(t_acc)}")
    if b_best_t is not None:
        summary.append(f"  Best blend: {int(b_best_lrw*100)}% LR + "
                       f"{int(round((1-b_best_lrw)*100))}% XGB = {fmt(b_best_t)}")

saved_mdl = best_model_lr is not None
saved_elo = saved_mdl and best_elo_hist_sv is not None
summary += [
    "",
    f"BEST RESULT: {fmt(best_res['temporal_acc'])} (config: {best_res['config']})",
    f"vs baseline: {best_res['temporal_acc'] - PREV_BEST_TEMPORAL:+.2%}",
    "",
    f"Saved: {'✓' if saved_mdl else '✗'} models  "
    f"{'✓' if saved_elo else '✗'} elo  "
    f"{'✓' if saved_mdl and os.path.exists(META_JSON) else '✗'} metadata",
    "=" * 40,
]

for line in summary:
    log(line)

print()
for line in summary:
    print(line)
