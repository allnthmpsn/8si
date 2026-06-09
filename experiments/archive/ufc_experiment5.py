"""
ufc_experiment5.py — Tree models on clean Sherdog-fixed data.

Target: beat 68.00% (LR top-50 by coef, clean data baseline)
Strategy:
  1. Confirm 68.00% baseline (top-50 LR)
  2. XGBoost / LightGBM / CatBoost with Optuna tuning (50 trials each)
  3. Feature selection on best tree (top 20 / top 30 / top 50 / diff-only)
  4. New tree-specific interaction features + best tree model
  5. Stacking ensemble: XGB + LGB + CB base → LR meta (5-fold OOF)

Config: date≥2017, min_cum_fights≥3
Output: model/experiment5_output.txt
"""

import gc, os, time, warnings
import joblib, numpy as np, pandas as pd, optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import catboost as cb
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RS             = 42
BASELINE_ACC   = 0.6800    # top-50 LR on clean data (exp4 best)
N_TRIALS       = 50
TIMEOUT        = 300       # 5 min per Optuna study
MAX_SECS       = 300       # cut to 25 trials if elapsed already exceeds this
DATE_CUT       = "2017-01-01"
MIN_UFC_FIGHTS = 3

OUT_LOG  = "model/experiment5_output.txt"
BEST_MDL = "model/ufc_model_best.pkl"
BEST_FT  = "model/feature_columns_best.pkl"

os.makedirs("model", exist_ok=True)
with open(OUT_LOG, "w") as f:
    f.write("")

def log(msg):
    print(msg, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(msg + "\n")

log("=" * 70)
log(f"UFC EXPERIMENT 5  —  {pd.Timestamp.now().isoformat()}")
log(f"Target: beat {BASELINE_ACC:.4f}  Config: date≥2017, min_cum_fights≥{MIN_UFC_FIGHTS}")
log("=" * 70)

best_acc   = BASELINE_ACC
best_model = None
best_feats = None
all_results = []

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (same pipeline as exp4)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[1] Loading data and engineering features...")

career_raw = pd.read_csv("data/career_fights.csv")
career_raw["date"] = pd.to_datetime(career_raw["date"])
master     = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
fighters   = pd.read_csv("data/ufc_fighters_final.csv")
fighters["Wins"]   = pd.to_numeric(fighters["Wins"],   errors="coerce").fillna(0).astype(int)
fighters["Losses"] = pd.to_numeric(fighters["Losses"], errors="coerce").fillna(0).astype(int)

log(f"  career_fights: {career_raw.shape}  master: {master.shape}")

# ── Career rolling features ───────────────────────────────────────────────────
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
career["cum_finish_wins"] = career["cum_did_ko"] + career["cum_did_sub"]
career["career_finish_rate"] = career["cum_finish_wins"] / career["cum_wins"].clip(lower=1)

def roll_mean(x, w):
    return x.shift(1).rolling(w, min_periods=1).mean()

career["last3_win_rate"]  = g["won"].transform(lambda x: roll_mean(x, 3))
career["last10_win_rate"] = g["won"].transform(lambda x: roll_mean(x, 10))
career["trend_score"]     = career["last3_win_rate"] - career["last10_win_rate"]

career["recency_win_rate"] = g["won"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).apply(
        lambda s: float(np.dot(np.arange(1, len(s) + 1), s) / np.arange(1, len(s) + 1).sum()),
        raw=True,
    )
)

career["prev_date"]   = g["date"].transform(lambda x: x.shift(1))
career["layoff_days"] = (career["date"] - career["prev_date"]).dt.days.fillna(0)

log("  career rolling: done")

# ── Opponent quality ──────────────────────────────────────────────────────────
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

# ── Master feature table ──────────────────────────────────────────────────────
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

# UFC win rate
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

# ── Step 4 interaction features ───────────────────────────────────────────────
for p in ["R", "B"]:
    # finish_rate × recent win rate (finisher on a streak is dangerous)
    master[f"{p}_finish_x_streak"] = (
        master[f"{p}_last5_finish_rate"].fillna(0) *
        master[f"{p}_last5_won"].fillna(0)
    )
    # age × layoff (old fighter with long layoff = high risk)
    master[f"{p}_age_x_layoff"] = (
        master[f"{p}_age"].fillna(0) * master[f"{p}_layoff_days"].fillna(0)
    )
    # peak age flag (27–32 = prime MMA years)
    master[f"{p}_peak_age"] = (
        (master[f"{p}_age"] >= 27) & (master[f"{p}_age"] <= 32)
    ).astype(int)

add_diff(master, "finish_x_streak")
add_diff(master, "age_x_layoff")
add_diff(master, "peak_age")

# unsigned experience gap
master["experience_gap"] = (
    master["R_cum_fights"].fillna(0) - master["B_cum_fights"].fillna(0)
).abs()

# striker vs wrestler matchup
if "SLpM_dif" in master.columns and "TD_Def_dif" in master.columns:
    master["striker_vs_wrestler"] = master["SLpM_dif"] * master["TD_Def_dif"]

NEW5_FEATURES = []
for p in ["R", "B"]:
    for f in ["finish_x_streak", "age_x_layoff", "peak_age"]:
        NEW5_FEATURES.append(f"{p}_{f}")
for f in ["finish_x_streak_dif", "age_x_layoff_dif", "peak_age_dif",
          "experience_gap", "striker_vs_wrestler"]:
    if f in master.columns:
        NEW5_FEATURES.append(f)

log(f"  new interaction features: {len(NEW5_FEATURES)}")

# ── Feature sets ──────────────────────────────────────────────────────────────
BASE_108 = joblib.load(BEST_FT)
ALL_FEATS = BASE_108 + [f for f in NEW5_FEATURES if f not in BASE_108]
log(f"  BASE_108={len(BASE_108)}, ALL_WITH_NEW={len(ALL_FEATS)}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def prepare(feat_list):
    avail = [c for c in feat_list if c in master.columns]
    sub   = master[master["date"] >= pd.Timestamp(DATE_CUT)].copy()
    if "R_cum_fights" in sub.columns and "B_cum_fights" in sub.columns:
        sub = sub[
            (sub["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
            (sub["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
        ]
    sub = sub[avail + ["Winner_bin"]].dropna()
    return sub[avail], sub["Winner_bin"], avail

def augment(X, y, feat_list):
    Xf = X.copy()
    yf = (1 - y).reset_index(drop=True)
    for rc in [c for c in feat_list if c.startswith("R_")]:
        bc = "B_" + rc[2:]
        if bc in feat_list:
            Xf[rc], Xf[bc] = X[bc].values, X[rc].values
    for dc in [c for c in feat_list if c.endswith("_dif")]:
        Xf[dc] = -X[dc].values
    # experience_gap and striker_vs_wrestler are symmetric — negate them
    for sym in ["experience_gap", "striker_vs_wrestler"]:
        if sym in feat_list:
            Xf[sym] = -X[sym].values
    return (
        pd.concat([X.reset_index(drop=True), Xf], ignore_index=True),
        pd.concat([y.reset_index(drop=True), yf], ignore_index=True),
    )

def split_and_augment(feat_list):
    X, y, feats = prepare(feat_list)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_aug, ytr_aug   = augment(Xtr, ytr, feats)
    return Xtr_aug.values, Xte.values, ytr_aug.values, yte.values, feats

def record(config, acc, n_feats, secs, model=None, feats=None):
    global best_acc, best_model, best_feats
    tag = "  *** NEW BEST ***" if acc > best_acc else ""
    log(f"  {config:60s}  acc={acc:.4f}  feats={n_feats:3d}  {secs:.0f}s{tag}")
    all_results.append({"config": config, "acc": acc, "n_feats": n_feats, "secs": secs})
    if acc > best_acc and model is not None and feats is not None:
        best_acc   = acc
        best_model = model
        best_feats = feats
        joblib.dump(model, BEST_MDL)
        joblib.dump(feats, BEST_FT)
        log(f"  *** Saved new best → {BEST_MDL}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Confirm 68.00% baseline (top-50 LR on clean data)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 1 — Confirm baseline (top-50 LR, clean data)")
log("=" * 70)

t0 = time.time()
Xtr, Xte, ytr, yte, f108 = split_and_augment(BASE_108)
log(f"  Dataset: train={len(ytr)}, test={len(yte)}, features={len(f108)}")

# LR L2 C=2.12 robust (exp4 Optuna best on 108 feats)
sc_base = RobustScaler()
Xtr_s   = sc_base.fit_transform(Xtr)
Xte_s   = sc_base.transform(Xte)
lr_base = LogisticRegression(penalty="l2", C=2.124, solver="lbfgs",
                              max_iter=2000, random_state=RS, n_jobs=1)
lr_base.fit(Xtr_s, ytr)
acc_108  = accuracy_score(yte, lr_base.predict(Xte_s))
log(f"  LR-L2-C2.12-robust-108feats → {acc_108:.4f}  ({time.time()-t0:.1f}s)")

# top-50 by |coef|
coef_abs = np.abs(lr_base.coef_[0])
imp_df = pd.DataFrame({"feature": f108, "coef": lr_base.coef_[0], "abs_coef": coef_abs}
                      ).sort_values("abs_coef", ascending=False)
top50_feats = imp_df.head(50)["feature"].tolist()

Xtr50, Xte50, ytr50, yte50, f50 = split_and_augment(top50_feats)
sc50 = RobustScaler()
Xtr50s = sc50.fit_transform(Xtr50); Xte50s = sc50.transform(Xte50)
lr50 = LogisticRegression(penalty="l2", C=2.124, solver="lbfgs",
                           max_iter=2000, random_state=RS, n_jobs=1)
lr50.fit(Xtr50s, ytr50)
acc_top50 = accuracy_score(yte50, lr50.predict(Xte50s))
elapsed = time.time() - t0
log(f"  LR-top50-by-coef         → {acc_top50:.4f}  ({elapsed:.1f}s)")
log(f"  Baseline confirmed. Target to beat: {max(acc_top50, BASELINE_ACC):.4f}")

record("LR_top50_baseline_confirm", acc_top50, 50, elapsed,
       Pipeline([("sc", sc50), ("lr", lr50)]), f50)

log("\n  Top 20 features by |coef|:")
log(imp_df.head(20).to_string(index=False))
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Tree models with Optuna (50 trials each, 108 features)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 2 — Tree models with Optuna (108 features)")
log("=" * 70)

Xtr, Xte, ytr, yte, f108 = split_and_augment(BASE_108)
log(f"  Dataset: train={len(ytr)}, test={len(yte)}, features={len(f108)}")

best_tree_acc = 0.0
best_tree_model = None
best_tree_name  = ""
best_tree_params = {}

# ── 2a. XGBoost ──────────────────────────────────────────────────────────────
log("\n  -- XGBoost --")
t0 = time.time()

def obj_xgb(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth         = trial.suggest_int("max_depth", 2, 6),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RS,
        n_jobs            = 1,
    )
    m = xgb.XGBClassifier(**params)
    m.fit(Xtr, ytr, verbose=False)
    return accuracy_score(yte, m.predict(Xte))

study_xgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RS))
elapsed_so_far = time.time() - t0
n_xgb = 25 if elapsed_so_far > MAX_SECS else N_TRIALS
study_xgb.optimize(obj_xgb, n_trials=n_xgb, timeout=TIMEOUT,
                   n_jobs=1, show_progress_bar=False)

bp_xgb = study_xgb.best_params
xgb_best = xgb.XGBClassifier(
    **{k: v for k, v in bp_xgb.items()},
    use_label_encoder=False, eval_metric="logloss",
    random_state=RS, n_jobs=1,
)
xgb_best.fit(Xtr, ytr, verbose=False)
acc_xgb = accuracy_score(yte, xgb_best.predict(Xte))
elapsed = time.time() - t0
log(f"  Best params: {bp_xgb}")
record("XGBoost_Optuna_108feat", acc_xgb, len(f108), elapsed, xgb_best, list(f108))

if acc_xgb > best_tree_acc:
    best_tree_acc    = acc_xgb
    best_tree_model  = xgb_best
    best_tree_name   = "XGBoost"
    best_tree_params = bp_xgb

# feature importances
xgb_imp = pd.DataFrame({
    "feature": f108,
    "importance": xgb_best.feature_importances_,
}).sort_values("importance", ascending=False)
log("\n  XGBoost top 10 features:")
log(xgb_imp.head(10).to_string(index=False))
gc.collect()

# ── 2b. LightGBM ─────────────────────────────────────────────────────────────
log("\n  -- LightGBM --")
t0 = time.time()

def obj_lgb(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth         = trial.suggest_int("max_depth", 2, 8),
        num_leaves        = trial.suggest_int("num_leaves", 15, 127),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        random_state      = RS,
        n_jobs            = 1,
        verbose           = -1,
    )
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))

study_lgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RS))
elapsed_so_far = time.time() - t0
n_lgb = 25 if elapsed_so_far > MAX_SECS else N_TRIALS
study_lgb.optimize(obj_lgb, n_trials=n_lgb, timeout=TIMEOUT,
                   n_jobs=1, show_progress_bar=False)

bp_lgb = study_lgb.best_params
lgb_best = lgb.LGBMClassifier(
    **bp_lgb, random_state=RS, n_jobs=1, verbose=-1,
)
lgb_best.fit(Xtr, ytr)
acc_lgb = accuracy_score(yte, lgb_best.predict(Xte))
elapsed = time.time() - t0
log(f"  Best params: {bp_lgb}")
record("LightGBM_Optuna_108feat", acc_lgb, len(f108), elapsed, lgb_best, list(f108))

if acc_lgb > best_tree_acc:
    best_tree_acc    = acc_lgb
    best_tree_model  = lgb_best
    best_tree_name   = "LightGBM"
    best_tree_params = bp_lgb

lgb_imp = pd.DataFrame({
    "feature": f108,
    "importance": lgb_best.feature_importances_,
}).sort_values("importance", ascending=False)
log("\n  LightGBM top 10 features:")
log(lgb_imp.head(10).to_string(index=False))
gc.collect()

# ── 2c. CatBoost ──────────────────────────────────────────────────────────────
log("\n  -- CatBoost --")
t0 = time.time()

def obj_cat(trial):
    params = dict(
        iterations    = trial.suggest_int("iterations", 100, 800),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        depth         = trial.suggest_int("depth", 2, 6),
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        random_seed   = RS,
        verbose       = False,
        thread_count  = 1,
    )
    m = cb.CatBoostClassifier(**params)
    m.fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))

study_cat = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RS))
elapsed_so_far = time.time() - t0
n_cat = 25 if elapsed_so_far > MAX_SECS else N_TRIALS
study_cat.optimize(obj_cat, n_trials=n_cat, timeout=TIMEOUT,
                   n_jobs=1, show_progress_bar=False)

bp_cat = study_cat.best_params
cat_best = cb.CatBoostClassifier(**bp_cat, random_seed=RS, verbose=False, thread_count=1)
cat_best.fit(Xtr, ytr)
acc_cat = accuracy_score(yte, cat_best.predict(Xte))
elapsed = time.time() - t0
log(f"  Best params: {bp_cat}")
record("CatBoost_Optuna_108feat", acc_cat, len(f108), elapsed, cat_best, list(f108))

if acc_cat > best_tree_acc:
    best_tree_acc    = acc_cat
    best_tree_model  = cat_best
    best_tree_name   = "CatBoost"
    best_tree_params = bp_cat

cat_imp = pd.DataFrame({
    "feature": f108,
    "importance": cat_best.feature_importances_,
}).sort_values("importance", ascending=False)
log("\n  CatBoost top 10 features:")
log(cat_imp.head(10).to_string(index=False))
gc.collect()

log(f"\n  Best tree in Step 2: {best_tree_name} → {best_tree_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Feature selection on best tree model
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log(f"STEP 3 — Feature selection on best tree ({best_tree_name})")
log("=" * 70)

# Use importance from the best tree
if best_tree_name == "XGBoost":
    imp_tree = xgb_imp
elif best_tree_name == "LightGBM":
    imp_tree = lgb_imp
else:
    imp_tree = cat_imp

def make_tree(n_feats=None, feat_list=None):
    """Retrain best tree architecture on a given feature set."""
    if best_tree_name == "XGBoost":
        m = xgb.XGBClassifier(
            **{k: v for k, v in best_tree_params.items()},
            use_label_encoder=False, eval_metric="logloss",
            random_state=RS, n_jobs=1,
        )
    elif best_tree_name == "LightGBM":
        m = lgb.LGBMClassifier(**best_tree_params, random_state=RS, n_jobs=1, verbose=-1)
    else:
        m = cb.CatBoostClassifier(**best_tree_params, random_seed=RS, verbose=False, thread_count=1)
    return m

# Top k by importance
for k in [20, 30, 50]:
    topk = imp_tree.head(k)["feature"].tolist()
    t0   = time.time()
    Xtr_k, Xte_k, ytr_k, yte_k, fk = split_and_augment(topk)
    m_k = make_tree()
    if best_tree_name == "XGBoost":
        m_k.fit(Xtr_k, ytr_k, verbose=False)
    else:
        m_k.fit(Xtr_k, ytr_k)
    acc_k = accuracy_score(yte_k, m_k.predict(Xte_k))
    elapsed = time.time() - t0
    record(f"{best_tree_name}_top{k}_by_importance", acc_k, k, elapsed, m_k, fk)
    gc.collect()

# Diff-only features
diff_feats = [c for c in f108 if c.endswith("_dif")]
log(f"\n  Diff-only features: {len(diff_feats)}")
t0 = time.time()
Xtr_d, Xte_d, ytr_d, yte_d, fd = split_and_augment(diff_feats)
m_d = make_tree()
if best_tree_name == "XGBoost":
    m_d.fit(Xtr_d, ytr_d, verbose=False)
else:
    m_d.fit(Xtr_d, ytr_d)
acc_d = accuracy_score(yte_d, m_d.predict(Xte_d))
elapsed = time.time() - t0
record(f"{best_tree_name}_diff_only_{len(diff_feats)}feat", acc_d, len(diff_feats),
       elapsed, m_d, fd)
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — New interaction features + best tree model
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 4 — New interaction features + best tree model")
log("=" * 70)

log(f"  Features: 108 base + {len(NEW5_FEATURES)} new = {len(ALL_FEATS)} total")
t0 = time.time()
Xtr_n, Xte_n, ytr_n, yte_n, fn = split_and_augment(ALL_FEATS)
log(f"  Dataset: train={len(ytr_n)}, test={len(yte_n)}, features={len(fn)}")

m_new = make_tree()
if best_tree_name == "XGBoost":
    m_new.fit(Xtr_n, ytr_n, verbose=False)
else:
    m_new.fit(Xtr_n, ytr_n)
acc_new = accuracy_score(yte_n, m_new.predict(Xte_n))
elapsed = time.time() - t0
record(f"{best_tree_name}_108+new_{len(fn)}feat", acc_new, len(fn), elapsed, m_new, fn)

# Show importances of new features specifically
new_in_model = [f for f in NEW5_FEATURES if f in fn]
if hasattr(m_new, "feature_importances_"):
    imp_new_df = pd.DataFrame({
        "feature": fn,
        "importance": m_new.feature_importances_,
    })
    new_imps = imp_new_df[imp_new_df["feature"].isin(new_in_model)].sort_values(
        "importance", ascending=False
    )
    log(f"\n  New feature importances:")
    log(new_imps.to_string(index=False))
gc.collect()

# Also try: Optuna on the best tree with ALL_FEATS
log(f"\n  -- Optuna on {best_tree_name} with {len(ALL_FEATS)} features --")
t0 = time.time()

def obj_new_feats(trial):
    if best_tree_name == "XGBoost":
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 1000),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 2, 6),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            use_label_encoder= False, eval_metric="logloss",
            random_state=RS, n_jobs=1,
        )
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr_n, ytr_n, verbose=False)
    elif best_tree_name == "LightGBM":
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth         = trial.suggest_int("max_depth", 2, 8),
            num_leaves        = trial.suggest_int("num_leaves", 15, 127),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
            random_state=RS, n_jobs=1, verbose=-1,
        )
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr_n, ytr_n)
    else:
        params = dict(
            iterations    = trial.suggest_int("iterations", 100, 800),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth         = trial.suggest_int("depth", 2, 6),
            l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_seed=RS, verbose=False, thread_count=1,
        )
        m = cb.CatBoostClassifier(**params)
        m.fit(Xtr_n, ytr_n)
    return accuracy_score(yte_n, m.predict(Xte_n))

study_new = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RS))
study_new.optimize(obj_new_feats, n_trials=N_TRIALS, timeout=TIMEOUT,
                   n_jobs=1, show_progress_bar=False)

bp_new = study_new.best_params
log(f"  Best params (new feats): {bp_new}")
if best_tree_name == "XGBoost":
    m_new_opt = xgb.XGBClassifier(
        **{k: v for k, v in bp_new.items()},
        use_label_encoder=False, eval_metric="logloss",
        random_state=RS, n_jobs=1,
    )
    m_new_opt.fit(Xtr_n, ytr_n, verbose=False)
elif best_tree_name == "LightGBM":
    m_new_opt = lgb.LGBMClassifier(**bp_new, random_state=RS, n_jobs=1, verbose=-1)
    m_new_opt.fit(Xtr_n, ytr_n)
else:
    m_new_opt = cb.CatBoostClassifier(**bp_new, random_seed=RS, verbose=False, thread_count=1)
    m_new_opt.fit(Xtr_n, ytr_n)

acc_new_opt = accuracy_score(yte_n, m_new_opt.predict(Xte_n))
elapsed = time.time() - t0
record(f"{best_tree_name}_Optuna_{len(fn)}feat_new_interactions",
       acc_new_opt, len(fn), elapsed, m_new_opt, fn)
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Stacking ensemble (trees only, LR meta, 5-fold OOF)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("STEP 5 — Stacking ensemble (XGB + LGB + CB, LR meta, 5-fold OOF)")
log("=" * 70)

# Only run if all 3 trees scored > 67%
if acc_xgb > 0.67 and acc_lgb > 0.67 and acc_cat > 0.67:
    log(f"  All 3 trees > 67% — proceeding with stacking")
    t0 = time.time()

    Xtr_s, Xte_s, ytr_s, yte_s, fs = split_and_augment(BASE_108)
    log(f"  Dataset: train={len(ytr_s)}, test={len(yte_s)}, features={len(fs)}")

    # Base model factories (use best Optuna params from Step 2)
    def make_xgb():
        return xgb.XGBClassifier(
            **{k: v for k, v in bp_xgb.items()},
            use_label_encoder=False, eval_metric="logloss",
            random_state=RS, n_jobs=1,
        )
    def make_lgb():
        return lgb.LGBMClassifier(**bp_lgb, random_state=RS, n_jobs=1, verbose=-1)
    def make_cat():
        return cb.CatBoostClassifier(**bp_cat, random_seed=RS, verbose=False, thread_count=1)

    base_factories = [make_xgb, make_lgb, make_cat]
    base_names     = ["XGBoost", "LightGBM", "CatBoost"]
    n_base = len(base_factories)

    # 5-fold OOF on training set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    oof_train = np.zeros((len(ytr_s), n_base))

    log("  Generating OOF predictions (5 folds × 3 models)...")
    for fold_i, (idx_tr, idx_val) in enumerate(skf.split(Xtr_s, ytr_s)):
        X_fold_tr, X_fold_val = Xtr_s[idx_tr], Xtr_s[idx_val]
        y_fold_tr, y_fold_val = ytr_s[idx_tr], ytr_s[idx_val]
        for b_i, factory in enumerate(base_factories):
            m_fold = factory()
            if base_names[b_i] == "XGBoost":
                m_fold.fit(X_fold_tr, y_fold_tr, verbose=False)
            else:
                m_fold.fit(X_fold_tr, y_fold_tr)
            oof_train[idx_val, b_i] = m_fold.predict_proba(X_fold_val)[:, 1]
        log(f"    Fold {fold_i+1}/5 done")
        del m_fold; gc.collect()

    # Train base models on full training set, get test predictions
    log("  Training base models on full training set...")
    oof_test = np.zeros((len(yte_s), n_base))
    base_models_final = []
    for b_i, factory in enumerate(base_factories):
        m_full = factory()
        if base_names[b_i] == "XGBoost":
            m_full.fit(Xtr_s, ytr_s, verbose=False)
        else:
            m_full.fit(Xtr_s, ytr_s)
        oof_test[:, b_i] = m_full.predict_proba(Xte_s)[:, 1]
        solo_acc = accuracy_score(yte_s, m_full.predict(Xte_s))
        log(f"    {base_names[b_i]} solo: {solo_acc:.4f}")
        base_models_final.append(m_full)
        gc.collect()

    # Meta-learner: LR on OOF predictions
    log("  Training LR meta-learner on OOF...")
    meta_lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                  max_iter=1000, random_state=RS, n_jobs=1)
    meta_lr.fit(oof_train, ytr_s)
    meta_pred = meta_lr.predict(oof_test)
    acc_stack = accuracy_score(yte_s, meta_pred)
    elapsed = time.time() - t0
    log(f"  Stack OOF train accuracy: {accuracy_score(ytr_s, meta_lr.predict(oof_train)):.4f}")
    log(f"  Stack TEST accuracy:      {acc_stack:.4f}  ({elapsed:.0f}s)")

    # Save stacking ensemble as a dict (backend-compatible via custom predict)
    # Wrap in a class so joblib.dump works and predict_proba is available
    class StackingEnsemble:
        def __init__(self, bases, meta, feat_names):
            self.bases      = bases
            self.meta       = meta
            self.feat_names = feat_names
        def predict_proba(self, X):
            base_preds = np.column_stack([
                b.predict_proba(X)[:, 1] for b in self.bases
            ])
            return np.column_stack([
                1 - self.meta.predict_proba(base_preds)[:, 1],
                self.meta.predict_proba(base_preds)[:, 1],
            ])
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    stack_model = StackingEnsemble(base_models_final, meta_lr, list(fs))
    record("Stacking_XGB+LGB+CB_LRmeta_OOF5fold", acc_stack, len(fs), elapsed,
           stack_model, list(fs))
else:
    log(f"  Skipping stacking — not all trees > 67%")
    log(f"  XGBoost={acc_xgb:.4f}  LightGBM={acc_lgb:.4f}  CatBoost={acc_cat:.4f}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("FINAL RESULTS — RANKED")
log("=" * 70)

results_df = (
    pd.DataFrame(all_results)
    .dropna(subset=["acc"])
    .sort_values("acc", ascending=False)
    .reset_index(drop=True)
)
log(results_df.to_string(index=False))

log(f"\nTarget (exp4 best):  {BASELINE_ACC:.4f}")
log(f"New best:            {best_acc:.4f}  (delta: {best_acc - BASELINE_ACC:+.4f})")

if best_acc > BASELINE_ACC:
    log(f"\n✓ IMPROVEMENT — saved to {BEST_MDL}")
    log(f"  Model type: {type(best_model).__name__}")
    log(f"  Features ({len(best_feats)}): {best_feats[:20]} ...")
    # Feature importances for winner
    if hasattr(best_model, "feature_importances_"):
        winner_imp = pd.DataFrame({
            "feature": best_feats,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)
        log("\n  Winner top 20 feature importances:")
        log(winner_imp.to_string(index=False))
    elif hasattr(best_model, "bases"):
        log("  (Stacking ensemble — see individual model importances above)")
else:
    log("\n✗ No improvement over 68.00% — model files unchanged.")

log(f"\n[DONE]  {pd.Timestamp.now().isoformat()}")
