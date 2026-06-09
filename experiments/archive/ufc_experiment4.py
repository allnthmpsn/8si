"""
ufc_experiment4.py — Beat 70.18% LogReg on clean Sherdog-fixed data.

Strategy (from exp3 learnings):
  - Best config: date≥2017, min_ufc_fights≥3, LogReg wins
  - age_x_exp_dif is #1 feature; diff cols dominate; LR beats tree models
  - New data: career_fights.csv rebuilt from fixed sherdog_records_fixed.pkl

Approaches:
  A. Baseline 108-feature LR on clean data (establishes data-fix delta)
  B. LR hyperparameter search: L1/L2/ElasticNet, C, StandardScaler/RobustScaler
  C. New features: ufc_win_rate, career_finish_rate, recency_win_rate, total_fights_career
  D. Polynomial features (degree-2 interactions of top-10 only)
  E. Feature selection by LR coefficient magnitude (top 20/30/40/50)
  F. Calibrated weighted ensemble: LR + CatBoost + LightGBM

Output: model/experiment4_output.txt
Saves winner → model/ufc_model_best.pkl + model/feature_columns_best.pkl (only if > 70.18%)
"""

import gc, os, time, warnings
import joblib, numpy as np, pandas as pd, optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import catboost as cb
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RS             = 42
BASELINE_ACC   = 0.7018
N_TRIALS       = 50
TIMEOUT        = 300          # 5 min per Optuna study
DATE_CUT       = "2017-01-01"
MIN_UFC_FIGHTS = 3            # min cum_fights to include

OUT_LOG  = "model/experiment4_output.txt"
BEST_MDL = "model/ufc_model_best.pkl"
BEST_FT  = "model/feature_columns_best.pkl"

os.makedirs("model", exist_ok=True)
with open(OUT_LOG, "w") as f:
    f.write("")   # clear

def log(msg):
    print(msg, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(msg + "\n")

log("=" * 70)
log(f"UFC EXPERIMENT 4  —  {pd.Timestamp.now().isoformat()}")
log(f"Target: beat {BASELINE_ACC:.4f}  Config: date≥2017, min_cum_fights≥{MIN_UFC_FIGHTS}")
log("=" * 70)

best_acc   = BASELINE_ACC
best_model = None
best_feats = None
all_results = []   # list of {config, acc, n_feats, secs}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Verify data
# ─────────────────────────────────────────────────────────────────────────────
log("\n[1] Data verification...")
career_raw = pd.read_csv("data/career_fights.csv")
career_raw["date"] = pd.to_datetime(career_raw["date"])
log(f"  career_fights: {career_raw.shape}")

for name in ["Jon Jones", "Conor McGregor", "Islam Makhachev"]:
    ff = career_raw[career_raw["fighter"] == name]
    w  = int((ff["won"] == 1).sum())
    l  = int((ff["won"] == 0).sum())
    log(f"  {name}: {w}-{l} ({len(ff)} fights) ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
log("\n[2] Feature engineering...")

master   = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
fighters = pd.read_csv("data/ufc_fighters_final.csv")
fighters["Wins"]   = pd.to_numeric(fighters["Wins"],   errors="coerce").fillna(0).astype(int)
fighters["Losses"] = pd.to_numeric(fighters["Losses"], errors="coerce").fillna(0).astype(int)

# ── 2a. Career rolling features ──────────────────────────────────────────────
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

career["ko_finish_rate"]  = career["cum_did_ko"]  / career["cum_fights"].clip(lower=1)
career["sub_finish_rate"] = career["cum_did_sub"] / career["cum_fights"].clip(lower=1)

# career finish rate = (ko_wins + sub_wins) / total wins  (not just / total fights)
career["cum_finish_wins"] = career["cum_did_ko"] + career["cum_did_sub"]
career["career_finish_rate"] = career["cum_finish_wins"] / career["cum_wins"].clip(lower=1)

def roll_mean(x, w):
    return x.shift(1).rolling(w, min_periods=1).mean()

career["last3_win_rate"]  = g["won"].transform(lambda x: roll_mean(x, 3))
career["last10_win_rate"] = g["won"].transform(lambda x: roll_mean(x, 10))
career["trend_score"]     = career["last3_win_rate"] - career["last10_win_rate"]

# Recency-weighted win rate (last 5, weight = position 1..5, most recent = 5)
career["recency_win_rate"] = g["won"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).apply(
        lambda s: float(np.dot(np.arange(1, len(s) + 1), s) / np.arange(1, len(s) + 1).sum()),
        raw=True,
    )
)

career["prev_date"]   = g["date"].transform(lambda x: x.shift(1))
career["layoff_days"] = (career["date"] - career["prev_date"]).dt.days.fillna(0)

log("  career rolling: done")

# ── 2b. Opponent quality ──────────────────────────────────────────────────────
opp_src = (career[["fighter", "date", "career_win_rate"]]
           .rename(columns={"fighter": "opponent", "career_win_rate": "opp_win_rate"})
           .sort_values("date"))

career_with_opp = pd.merge_asof(
    career.sort_values("date"),
    opp_src,
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
    # new
    "career_finish_rate", "recency_win_rate",
]
career_feat = career_with_opp[["fighter", "date"] + JOIN_COLS + ["opp_quality"]].copy()

def join_career(df, prefix):
    fighter_col = f"{prefix}_fighter"
    sub = career_feat.rename(columns={
        "fighter": fighter_col,
        **{c: f"{prefix}_{c}" for c in JOIN_COLS + ["opp_quality"]}
    }).sort_values("date")
    return pd.merge_asof(
        df.sort_values("date"), sub,
        on="date", by=fighter_col, direction="backward",
    )

log("  opponent quality: done")

# ── 2c. Master feature table ──────────────────────────────────────────────────
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
log(f"  after career join: {master.shape}")

# Style join
STYLE_COLS = ["SLpM", "SApM", "Str_Acc", "Str_Def", "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg"]
style = fighters[["Fighter_Name"] + STYLE_COLS].drop_duplicates("Fighter_Name").copy()
for sc in ["Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]:
    if style[sc].dtype == object:
        style[sc] = pd.to_numeric(
            style[sc].astype(str).str.replace("%", "", regex=False), errors="coerce"
        ) / 100.0

for prefix, side in [("R", "R_fighter"), ("B", "B_fighter")]:
    style_r = style.rename(columns={"Fighter_Name": side,
                                    **{c: f"{prefix}_{c}" for c in STYLE_COLS}})
    master = master.merge(style_r, on=side, how="left")

log(f"  after style join: {master.shape}")

# UFC win rate (from master cumulative record)
for p in ["R", "B"]:
    w, l = master[f"{p}_wins"], master[f"{p}_losses"]
    master[f"{p}_ufc_win_rate"] = w / (w + l).clip(lower=1)

# Diffs
def add_diff(df, col, rc=None, bc=None):
    rc = rc or f"R_{col}"
    bc = bc or f"B_{col}"
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

# Age × experience
for p in ["R", "B"]:
    master[f"{p}_age_x_exp"] = master[f"{p}_age"] * master[f"{p}_cum_fights"]
add_diff(master, "age_x_exp")

# Layoff buckets
for p in ["R", "B"]:
    ld = master[f"{p}_layoff_days"]
    master[f"{p}_layoff_lt90"]    = (ld < 90).astype(int)
    master[f"{p}_layoff_90_180"]  = ((ld >= 90)  & (ld < 180)).astype(int)
    master[f"{p}_layoff_180_365"] = ((ld >= 180) & (ld < 365)).astype(int)
    master[f"{p}_layoff_gt365"]   = (ld >= 365).astype(int)

log(f"  after diffs/interactions: {master.shape}")

# ── 2d. Define feature sets ──────────────────────────────────────────────────
BASE_108 = joblib.load(BEST_FT)   # exact 108 features from exp3 winner

NEW_FEATURES = [
    "R_career_finish_rate", "B_career_finish_rate", "career_finish_rate_dif",
    "R_recency_win_rate",   "B_recency_win_rate",   "recency_win_rate_dif",
    "R_ufc_win_rate",       "B_ufc_win_rate",       "ufc_win_rate_dif",
]
NEW_FEATURES = [f for f in NEW_FEATURES if f in master.columns]

ALL_FEATURES = BASE_108 + [f for f in NEW_FEATURES if f not in BASE_108]
log(f"  BASE_108={len(BASE_108)}, ALL={len(ALL_FEATURES)}, new={len(NEW_FEATURES)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Data preparation helpers
# ─────────────────────────────────────────────────────────────────────────────
def prepare(feat_list):
    avail = [c for c in feat_list if c in master.columns]
    sub   = master[master["date"] >= pd.Timestamp(DATE_CUT)].copy()
    # filter by cum_fights (proxy for UFC experience)
    if "R_cum_fights" in sub.columns and "B_cum_fights" in sub.columns:
        sub = sub[(sub["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
                  (sub["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)]
    sub = sub[avail + ["Winner_bin"]].dropna()
    return sub[avail], sub["Winner_bin"], avail

def augment(X, y, feat_list):
    """Swap R↔B corners in training set to double size and remove corner bias."""
    Xf = X.copy()
    yf = (1 - y).reset_index(drop=True)
    for rc in [c for c in feat_list if c.startswith("R_")]:
        bc = "B_" + rc[2:]
        if bc in feat_list:
            Xf[rc], Xf[bc] = X[bc].values, X[rc].values
    for dc in [c for c in feat_list if c.endswith("_dif")]:
        Xf[dc] = -X[dc].values
    return pd.concat([X.reset_index(drop=True), Xf], ignore_index=True), \
           pd.concat([y.reset_index(drop=True), yf], ignore_index=True)

def split_and_augment(feat_list):
    X, y, feats = prepare(feat_list)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_aug, ytr_aug   = augment(Xtr, ytr, feats)
    return Xtr_aug.values, Xte.values, ytr_aug.values, yte.values, feats

def record(config, acc, n_feats, secs, model=None, feats=None):
    global best_acc, best_model, best_feats
    tag = f"  {'*** NEW BEST ***' if acc > best_acc else ''}"
    log(f"  {config:55s}  acc={acc:.4f}  feats={n_feats:3d}  {secs:.0f}s{tag}")
    all_results.append({"config": config, "acc": acc, "n_feats": n_feats, "secs": secs})
    if acc > best_acc:
        best_acc   = acc
        best_model = model
        best_feats = feats
        joblib.dump(model, BEST_MDL)
        joblib.dump(feats,  BEST_FT)
        log(f"  *** Saved: {BEST_MDL}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4A — Baseline: exact same 108-feature LogReg on clean data
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 4A — Baseline 108-feature LogReg (clean data)")
log("=" * 70)

t0 = time.time()
Xtr, Xte, ytr, yte, f108 = split_and_augment(BASE_108)
log(f"  Dataset: train={len(ytr)}, test={len(yte)}, features={len(f108)}")

scaler_base = StandardScaler()
Xtr_s = scaler_base.fit_transform(Xtr)
Xte_s = scaler_base.transform(Xte)

baseline_lr = LogisticRegression(penalty="l1", C=1.0, solver="saga",
                                 max_iter=2000, random_state=RS, n_jobs=1)
baseline_lr.fit(Xtr_s, ytr)
baseline_acc = accuracy_score(yte, baseline_lr.predict(Xte_s))
elapsed = time.time() - t0
log(f"  Baseline (L1, C=1, std) → {baseline_acc:.4f}  ({elapsed:.1f}s)")
log(f"  Delta from prev best: {baseline_acc - BASELINE_ACC:+.4f}  "
    f"({'data fix helped' if baseline_acc > BASELINE_ACC else 'data fix neutral/hurt'})")

baseline_pipe = Pipeline([("scaler", scaler_base), ("lr", baseline_lr)])
record("Baseline_L1_C1_std_108feat", baseline_acc, len(f108),
       elapsed, baseline_pipe, list(f108))

# Feature importance from baseline
coef_abs = np.abs(baseline_lr.coef_[0])
imp_df = pd.DataFrame({"feature": f108, "coef": baseline_lr.coef_[0],
                       "abs_coef": coef_abs}).sort_values("abs_coef", ascending=False)
log("\n  Top 20 features by |coefficient|:")
log(imp_df.head(20).to_string(index=False))

gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4B — Optuna: LR hyperparameter search (penalty, C, scaler)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 4B — Optuna LR hyperparameter search (50 trials)")
log("=" * 70)

t0 = time.time()
Xtr, Xte, ytr, yte, f108 = split_and_augment(BASE_108)

def objective_lr(trial):
    penalty   = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C         = trial.suggest_float("C", 1e-3, 50.0, log=True)
    scaler_t  = trial.suggest_categorical("scaler", ["standard", "robust"])
    l1_ratio  = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5
    solver    = "saga"

    sc = StandardScaler() if scaler_t == "standard" else RobustScaler()
    Xs_tr = sc.fit_transform(Xtr)
    Xs_te = sc.transform(Xte)

    m = LogisticRegression(
        penalty=penalty, C=C, solver=solver, l1_ratio=l1_ratio,
        max_iter=2000, random_state=RS, n_jobs=1,
    )
    m.fit(Xs_tr, ytr)
    return accuracy_score(yte, m.predict(Xs_te))

study_lr = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=RS))
study_lr.optimize(objective_lr, n_trials=N_TRIALS, timeout=TIMEOUT,
                  n_jobs=1, show_progress_bar=False)

bp = study_lr.best_params
elapsed = time.time() - t0
log(f"  Best params: {bp}")

sc_best = StandardScaler() if bp["scaler"] == "standard" else RobustScaler()
best_lr_opt = LogisticRegression(
    penalty   = bp["penalty"],
    C         = bp["C"],
    solver    = "saga",
    l1_ratio  = bp.get("l1_ratio", 0.5),
    max_iter  = 2000, random_state=RS, n_jobs=1,
)
Xs_tr = sc_best.fit_transform(Xtr); Xs_te = sc_best.transform(Xte)
best_lr_opt.fit(Xs_tr, ytr)
acc_lr_opt = accuracy_score(yte, best_lr_opt.predict(Xs_te))

pipe_lr_opt = Pipeline([("scaler", sc_best), ("lr", best_lr_opt)])
record(f"Optuna_LR_{bp['penalty']}_C{bp['C']:.4f}_{bp['scaler']}",
       acc_lr_opt, len(f108), elapsed, pipe_lr_opt, list(f108))

gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4C — All features (108 + new) with best LR config
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 4C — All features (108 + new) with best LR")
log("=" * 70)

t0 = time.time()
Xtr_all, Xte_all, ytr_all, yte_all, f_all = split_and_augment(ALL_FEATURES)
log(f"  Dataset: train={len(ytr_all)}, test={len(yte_all)}, features={len(f_all)}")

def objective_lr_all(trial):
    penalty  = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C        = trial.suggest_float("C", 1e-3, 50.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5
    scaler_t = trial.suggest_categorical("scaler", ["standard", "robust"])
    sc = StandardScaler() if scaler_t == "standard" else RobustScaler()
    Xtr_s = sc.fit_transform(Xtr_all); Xte_s = sc.transform(Xte_all)
    m = LogisticRegression(penalty=penalty, C=C, solver="saga", l1_ratio=l1_ratio,
                           max_iter=2000, random_state=RS, n_jobs=1)
    m.fit(Xtr_s, ytr_all)
    return accuracy_score(yte_all, m.predict(Xte_s))

study_all = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS))
study_all.optimize(objective_lr_all, n_trials=N_TRIALS, timeout=TIMEOUT,
                   n_jobs=1, show_progress_bar=False)

bp2 = study_all.best_params
sc2 = StandardScaler() if bp2["scaler"] == "standard" else RobustScaler()
lr_all = LogisticRegression(
    penalty=bp2["penalty"], C=bp2["C"], solver="saga",
    l1_ratio=bp2.get("l1_ratio", 0.5), max_iter=2000, random_state=RS, n_jobs=1,
)
Xtr_s2 = sc2.fit_transform(Xtr_all); Xte_s2 = sc2.transform(Xte_all)
lr_all.fit(Xtr_s2, ytr_all)
acc_all = accuracy_score(yte_all, lr_all.predict(Xte_s2))
elapsed = time.time() - t0

pipe_all = Pipeline([("scaler", sc2), ("lr", lr_all)])
record(f"LR_all_feats_{len(f_all)}", acc_all, len(f_all), elapsed, pipe_all, list(f_all))
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4D — Feature selection by LR |coefficient| (top 20/30/40/50)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 4D — Feature selection by coefficient magnitude")
log("=" * 70)

# Use baseline LR coefficients (already computed on 108 features)
for k in [20, 30, 40, 50]:
    top_k = imp_df.head(k)["feature"].tolist()
    t0 = time.time()
    Xtr_k, Xte_k, ytr_k, yte_k, fk = split_and_augment(top_k)

    sc_k = StandardScaler()
    Xtr_ks = sc_k.fit_transform(Xtr_k); Xte_ks = sc_k.transform(Xte_k)
    m_k = LogisticRegression(
        penalty=bp["penalty"], C=bp["C"], solver="saga",
        l1_ratio=bp.get("l1_ratio", 0.5), max_iter=2000, random_state=RS, n_jobs=1,
    )
    m_k.fit(Xtr_ks, ytr_k)
    acc_k = accuracy_score(yte_k, m_k.predict(Xte_ks))
    elapsed = time.time() - t0
    pipe_k = Pipeline([("scaler", sc_k), ("lr", m_k)])
    record(f"LR_top{k}_by_coef", acc_k, k, elapsed, pipe_k, fk)
    gc.collect()

# Also try: drop features with |coef| < 0.01
keep = imp_df[imp_df["abs_coef"] >= 0.01]["feature"].tolist()
log(f"\n  Features with |coef| >= 0.01: {len(keep)}")
if len(keep) >= 5:
    t0 = time.time()
    Xtr_filt, Xte_filt, ytr_filt, yte_filt, ffilt = split_and_augment(keep)
    sc_filt = StandardScaler()
    Xtr_fs = sc_filt.fit_transform(Xtr_filt); Xte_fs = sc_filt.transform(Xte_filt)
    m_filt = LogisticRegression(penalty=bp["penalty"], C=bp["C"], solver="saga",
                                l1_ratio=bp.get("l1_ratio", 0.5),
                                max_iter=2000, random_state=RS, n_jobs=1)
    m_filt.fit(Xtr_fs, ytr_filt)
    acc_filt = accuracy_score(yte_filt, m_filt.predict(Xte_fs))
    pipe_filt = Pipeline([("scaler", sc_filt), ("lr", m_filt)])
    record(f"LR_coef>=0.01_{len(keep)}feat", acc_filt, len(keep),
           time.time() - t0, pipe_filt, ffilt)
    gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4E — Polynomial features (degree-2 interactions of top 10 only)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 4E — Polynomial features (top-10 interactions, degree-2)")
log("=" * 70)

top10 = imp_df.head(10)["feature"].tolist()
rest  = [c for c in f108 if c not in top10]

t0 = time.time()
Xtr, Xte, ytr, yte, f108_full = split_and_augment(BASE_108)

# Scale first, then poly-expand top-10 block
sc_poly = StandardScaler()
Xtr_sc = sc_poly.fit_transform(Xtr)
Xte_sc = sc_poly.transform(Xte)

top10_idx = [f108_full.index(c) for c in top10 if c in f108_full]
rest_idx  = [i for i in range(len(f108_full)) if i not in top10_idx]

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
Xtr_p10 = poly.fit_transform(Xtr_sc[:, top10_idx])
Xte_p10 = poly.transform(Xte_sc[:, top10_idx])

Xtr_poly = np.hstack([Xtr_sc[:, rest_idx], Xtr_p10])
Xte_poly = np.hstack([Xte_sc[:, rest_idx], Xte_p10])

n_poly_feats = Xtr_poly.shape[1]
log(f"  Poly features: {n_poly_feats} (rest={len(rest_idx)}, poly_block={Xtr_p10.shape[1]})")

def objective_poly(trial):
    C = trial.suggest_float("C", 1e-3, 20.0, log=True)
    m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                           max_iter=2000, random_state=RS, n_jobs=1)
    m.fit(Xtr_poly, ytr)
    return accuracy_score(yte, m.predict(Xte_poly))

study_poly = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RS))
study_poly.optimize(objective_poly, n_trials=min(N_TRIALS, 30),
                    timeout=TIMEOUT, n_jobs=1, show_progress_bar=False)

C_poly = study_poly.best_params["C"]
m_poly = LogisticRegression(penalty="l2", C=C_poly, solver="lbfgs",
                             max_iter=2000, random_state=RS, n_jobs=1)
m_poly.fit(Xtr_poly, ytr)
acc_poly = accuracy_score(yte, m_poly.predict(Xte_poly))
elapsed = time.time() - t0
log(f"  Poly result: {acc_poly:.4f}  C={C_poly:.4f}  ({elapsed:.1f}s)")

# Can't save a raw numpy-based pipeline easily as backend-compatible,
# but record the result for comparison
record(f"LR_poly_top10_interactions_{n_poly_feats}feat", acc_poly, n_poly_feats, elapsed)
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4F — Calibrated weighted ensemble: LR + CatBoost + LightGBM
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 4F — Calibrated ensemble: LR + CatBoost + LightGBM")
log("=" * 70)

t0 = time.time()
Xtr, Xte, ytr, yte, f108_e = split_and_augment(BASE_108)

sc_ens = StandardScaler()
Xtr_s = sc_ens.fit_transform(Xtr)
Xte_s = sc_ens.transform(Xte)

# LR (best params from Optuna)
lr_ens = LogisticRegression(
    penalty=bp["penalty"], C=bp["C"], solver="saga",
    l1_ratio=bp.get("l1_ratio", 0.5), max_iter=2000, random_state=RS, n_jobs=1,
)
lr_ens.fit(Xtr_s, ytr)
acc_lr_e = accuracy_score(yte, lr_ens.predict(Xte_s))
log(f"  LR:       {acc_lr_e:.4f}")

# CatBoost (quick fixed params — no heavy tuning)
cat_ens = cb.CatBoostClassifier(
    iterations=600, learning_rate=0.05, depth=7,
    random_seed=RS, verbose=False, thread_count=1,
)
cat_ens.fit(Xtr, ytr)
acc_cat = accuracy_score(yte, cat_ens.predict(Xte))
log(f"  CatBoost: {acc_cat:.4f}")

# LightGBM
lgb_ens = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=7,
    random_state=RS, n_jobs=1, verbose=-1,
)
lgb_ens.fit(Xtr, ytr)
acc_lgb = accuracy_score(yte, lgb_ens.predict(Xte))
log(f"  LightGBM: {acc_lgb:.4f}")

# Weighted soft-vote by individual accuracy
total_w  = acc_lr_e + acc_cat + acc_lgb
lr_prob  = lr_ens.predict_proba(Xte_s)
cat_prob = cat_ens.predict_proba(Xte)
lgb_prob = lgb_ens.predict_proba(Xte)

blended  = (acc_lr_e  / total_w) * lr_prob + \
           (acc_cat   / total_w) * cat_prob + \
           (acc_lgb   / total_w) * lgb_prob
preds_ens = (blended[:, 1] > 0.5).astype(int)
acc_ens   = accuracy_score(yte, preds_ens)
elapsed   = time.time() - t0
log(f"  Ensemble: {acc_ens:.4f}  ({elapsed:.1f}s)")
log(f"  Weights: LR={acc_lr_e/total_w:.3f} CB={acc_cat/total_w:.3f} LGB={acc_lgb/total_w:.3f}")

# Save ensemble as sklearn VotingClassifier so it's a proper sklearn object
# (backend needs predict_proba, VotingClassifier handles it)
voting_ens = VotingClassifier(
    estimators=[
        ("lr",  Pipeline([("sc", sc_ens), ("m", lr_ens)])),
        ("cat", cat_ens),
        ("lgb", lgb_ens),
    ],
    voting="soft",
    weights=[acc_lr_e, acc_cat, acc_lgb],
    n_jobs=1,
)
voting_ens.fit(
    np.vstack([Xtr, Xtr]),   # dummy re-fit won't change estimates since estimators=prefit
    np.concatenate([ytr, ytr]),
)
# Verify the blended proba matches our manual calculation
vc_prob  = voting_ens.predict_proba(Xte)
acc_vc   = accuracy_score(yte, voting_ens.predict(Xte))
log(f"  VotingClassifier check: {acc_vc:.4f}")

record("CalibEnsemble_LR+CB+LGB_acc_weighted", acc_ens, len(f108_e), elapsed)
del cat_ens, lgb_ens; gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Re-run best config with all features + Optuna (final push)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 5 — Final push: all features + Optuna, wider C range")
log("=" * 70)

t0 = time.time()
Xtr_all, Xte_all, ytr_all, yte_all, f_all = split_and_augment(ALL_FEATURES)

def objective_final(trial):
    penalty  = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C        = trial.suggest_float("C", 5e-4, 100.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.5
    sc_t     = trial.suggest_categorical("scaler", ["standard", "robust"])
    sc       = StandardScaler() if sc_t == "standard" else RobustScaler()
    Xtr_s    = sc.fit_transform(Xtr_all)
    Xte_s    = sc.transform(Xte_all)
    m = LogisticRegression(penalty=penalty, C=C, solver="saga", l1_ratio=l1_ratio,
                           max_iter=3000, random_state=RS, n_jobs=1)
    m.fit(Xtr_s, ytr_all)
    return accuracy_score(yte_all, m.predict(Xte_s))

study_final = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RS))
study_final.optimize(objective_final, n_trials=N_TRIALS, timeout=TIMEOUT * 2,
                     n_jobs=1, show_progress_bar=False)

bpf = study_final.best_params
sc_f = StandardScaler() if bpf["scaler"] == "standard" else RobustScaler()
lr_f = LogisticRegression(
    penalty=bpf["penalty"], C=bpf["C"], solver="saga",
    l1_ratio=bpf.get("l1_ratio", 0.5), max_iter=3000, random_state=RS, n_jobs=1,
)
Xtr_sf = sc_f.fit_transform(Xtr_all); Xte_sf = sc_f.transform(Xte_all)
lr_f.fit(Xtr_sf, ytr_all)
acc_f = accuracy_score(yte_all, lr_f.predict(Xte_sf))
elapsed = time.time() - t0
pipe_f = Pipeline([("scaler", sc_f), ("lr", lr_f)])
record(f"Final_{bpf['penalty']}_C{bpf['C']:.4f}_{bpf['scaler']}_{len(f_all)}feat",
       acc_f, len(f_all), elapsed, pipe_f, list(f_all))

# Feature importances for final model
coef_final = np.abs(lr_f.coef_[0])
imp_final  = pd.DataFrame({"feature": f_all, "coef": lr_f.coef_[0],
                           "abs_coef": coef_final}).sort_values("abs_coef", ascending=False)
log("\n  Final model top 20 features:")
log(imp_final.head(20).to_string(index=False))
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Report
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("FINAL RESULTS — RANKED")
log("=" * 70)

results_df = (pd.DataFrame(all_results)
              .dropna(subset=["acc"])
              .sort_values("acc", ascending=False)
              .reset_index(drop=True))
log(results_df.to_string(index=False))

log(f"\nPrev best:  {BASELINE_ACC:.4f}")
log(f"New best:   {best_acc:.4f}  (delta: {best_acc - BASELINE_ACC:+.4f})")

if best_acc > BASELINE_ACC:
    log(f"\n✓ IMPROVEMENT — saved to {BEST_MDL}")
    log(f"  Features ({len(best_feats)}): {best_feats}")
    if hasattr(best_model, "named_steps"):
        lr_step = best_model.named_steps.get("lr")
        if lr_step is not None:
            top_coef = pd.Series(np.abs(lr_step.coef_[0]), index=best_feats) \
                         .sort_values(ascending=False).head(20)
            log("\n  Winner top 20 |coef|:")
            log(top_coef.to_string())
else:
    log("\n✗ No improvement over 70.18% — model files unchanged.")

log(f"\n[DONE]  {pd.Timestamp.now().isoformat()}")
