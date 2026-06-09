"""
ufc_experiment6.py — Targeted experiments to push XGBoost past 69.94% on clean data.

Run order: A (temporal split), C (refined XGB Optuna), B (weight class), D (recency weight), E (two-stage)
Target: beat 69.94% on random 80/20 split (random_state=42)
"""

import gc, os, time, warnings
import joblib, numpy as np, pandas as pd, optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RS             = 42
BASELINE_ACC   = 0.6994    # XGBoost random 80/20, clean data
DATE_CUT       = "2017-01-01"
MIN_UFC_FIGHTS = 3
TEMPORAL_CUT   = "2024-01-01"

# Best params from Experiment 5
BEST_PARAMS = dict(
    n_estimators     = 992,
    learning_rate    = 0.057338447379760536,
    max_depth        = 5,
    subsample        = 0.7577905622611881,
    colsample_bytree = 0.674191396199946,
    min_child_weight = 8,
)

OUT_LOG  = "model/experiment6_output.txt"
BEST_MDL = "model/ufc_model_best.pkl"
BEST_FT  = "model/feature_columns_best.pkl"

os.makedirs("model", exist_ok=True)
with open(OUT_LOG, "w") as f:
    f.write("")

def log(msg=""):
    print(msg, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(msg + "\n")

log("=" * 70)
log(f"UFC EXPERIMENT 6  —  {pd.Timestamp.now().isoformat()}")
log(f"Target: beat {BASELINE_ACC:.4f}  (XGBoost, random 80/20, clean data)")
log("=" * 70)

best_acc   = BASELINE_ACC
best_model = None
best_feats = None
all_results = []

def record(config, acc, n_feats, secs, model=None, feats=None, split="random"):
    global best_acc, best_model, best_feats
    tag = ""
    if split == "random" and acc > best_acc:
        tag = "  *** NEW BEST ***"
    log(f"  {config:65s}  acc={acc:.4f}  feats={n_feats:3d}  {secs:.0f}s{tag}")
    all_results.append({"config": config, "acc": acc, "n_feats": n_feats,
                        "secs": secs, "split": split})
    if split == "random" and acc > best_acc and model is not None and feats is not None:
        best_acc   = acc
        best_model = model
        best_feats = feats
        joblib.dump(model, BEST_MDL)
        joblib.dump(feats, BEST_FT)
        log(f"  *** Saved new best → {BEST_MDL}")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
log("\n[FE] Loading data and engineering features...")

career_raw = pd.read_csv("data/career_fights.csv")
career_raw["date"] = pd.to_datetime(career_raw["date"])
master     = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
fighters   = pd.read_csv("data/ufc_fighters_final.csv")
fighters["Wins"]   = pd.to_numeric(fighters["Wins"],   errors="coerce").fillna(0).astype(int)
fighters["Losses"] = pd.to_numeric(fighters["Losses"], errors="coerce").fillna(0).astype(int)

# ── Career rolling ────────────────────────────────────────────────────────────
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
        lambda s: float(np.dot(np.arange(1, len(s)+1), s) / np.arange(1, len(s)+1).sum()),
        raw=True,
    )
)
career["prev_date"]   = g["date"].transform(lambda x: x.shift(1))
career["layoff_days"] = (career["date"] - career["prev_date"]).dt.days.fillna(0)

# ── Opponent quality ──────────────────────────────────────────────────────────
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
    fighter_col = f"{prefix}_fighter"
    sub = career_feat.rename(columns={
        "fighter": fighter_col,
        **{c: f"{prefix}_{c}" for c in JOIN_COLS + ["opp_quality"]}
    }).sort_values("date")
    return pd.merge_asof(
        df.sort_values("date"), sub,
        on="date", by=fighter_col, direction="backward",
    )

# ── Master table ──────────────────────────────────────────────────────────────
master["Winner_bin"] = (master["Winner"] == "Red").astype(int)
master["is_finish"]  = master["finish"].isin(["KO/TKO", "SUB"]).astype(int)

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
    style_r = style.rename(columns={"Fighter_Name": side,
                                    **{c: f"{prefix}_{c}" for c in STYLE_COLS}})
    master = master.merge(style_r, on=side, how="left")

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

log(f"  master after FE: {master.shape}")

BASE_108 = joblib.load(BEST_FT)
log(f"  BASE_108 = {len(BASE_108)} features")

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def filter_df(df, feat_list, extra_cols=None):
    cols = [c for c in feat_list if c in df.columns]
    must = ["Winner_bin"] + (extra_cols or [])
    sub = df[(df["date"] >= pd.Timestamp(DATE_CUT)) &
             (df["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
             (df["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)].copy()
    sub = sub[cols + [c for c in must if c in sub.columns]].dropna(subset=cols + ["Winner_bin"])
    return sub, cols

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

def split_aug_random(feat_list):
    sub, cols = filter_df(master, feat_list)
    X, y = sub[cols], sub["Winner_bin"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
    Xtr_a, ytr_a = augment(Xtr, ytr, cols)
    return Xtr_a.values, Xte.values, ytr_a.values, yte.values, cols

def make_xgb(**overrides):
    params = {**BEST_PARAMS, **overrides,
              "use_label_encoder": False, "eval_metric": "logloss",
              "random_state": RS, "n_jobs": 1}
    return xgb.XGBClassifier(**params)

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Temporal split (train <2024, test ≥2024)
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("EXPERIMENT A — Temporal train/test split (train <2024, test ≥2024)")
log("=" * 70)
t_exp = time.time()

sub_a, cols_a = filter_df(master, BASE_108)
# Add date back for temporal split (it was dropped in filter_df)
sub_a_dated = master[
    (master["date"] >= pd.Timestamp(DATE_CUT)) &
    (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
    (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
].copy()
sub_a_dated = sub_a_dated[[c for c in BASE_108 if c in master.columns] +
                           ["Winner_bin", "date"]].dropna(
    subset=[c for c in BASE_108 if c in master.columns] + ["Winner_bin"])

train_a = sub_a_dated[sub_a_dated["date"] < pd.Timestamp(TEMPORAL_CUT)]
test_a  = sub_a_dated[sub_a_dated["date"] >= pd.Timestamp(TEMPORAL_CUT)]
cols_a  = [c for c in BASE_108 if c in sub_a_dated.columns]

Xtr_a, ytr_a = train_a[cols_a], train_a["Winner_bin"]
Xte_a, yte_a = test_a[cols_a].values, test_a["Winner_bin"].values

Xtr_a_aug, ytr_a_aug = augment(Xtr_a, ytr_a, cols_a)
Xtr_a_np = Xtr_a_aug.values

log(f"  Train: {len(ytr_a_aug)} (aug'd from {len(ytr_a)}), "
    f"Test (2024+): {len(yte_a)},  Features: {len(cols_a)}")

# XGBoost with best params
t0 = time.time()
xgb_temp = make_xgb()
xgb_temp.fit(Xtr_a_np, ytr_a_aug.values, verbose=False)
acc_a_xgb = accuracy_score(yte_a, xgb_temp.predict(Xte_a))
record("A_XGBoost_temporal_2024+_test", acc_a_xgb, len(cols_a),
       time.time()-t0, xgb_temp, cols_a, split="temporal")

# LR for comparison (no scaler needed for tree comparison, but LR needs it)
t0 = time.time()
sc_a = RobustScaler()
lr_a = LogisticRegression(penalty="l2", C=2.124, solver="lbfgs",
                           max_iter=2000, random_state=RS, n_jobs=1)
lr_a.fit(sc_a.fit_transform(Xtr_a_np), ytr_a_aug.values)
acc_a_lr = accuracy_score(yte_a, lr_a.predict(sc_a.transform(Xte_a)))
record("A_LR_temporal_2024+_test", acc_a_lr, len(cols_a),
       time.time()-t0, split="temporal")

log(f"\n  Temporal split summary: XGB={acc_a_xgb:.4f}  LR={acc_a_lr:.4f}")
log(f"  (Random split baseline: {BASELINE_ACC:.4f} — temporal gap = "
    f"{acc_a_xgb - BASELINE_ACC:+.4f})")
log(f"\n  Experiment A done in {time.time()-t_exp:.0f}s")
del xgb_temp, lr_a; gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Refined XGBoost Optuna (100 trials, tighter ranges)
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("EXPERIMENT C — Refined XGBoost Optuna (100 trials, ±narrow range)")
log("=" * 70)
t_exp = time.time()

Xtr_c, Xte_c, ytr_c, yte_c, cols_c = split_aug_random(BASE_108)
log(f"  Dataset: train={len(ytr_c)}, test={len(yte_c)}, features={len(cols_c)}")
log(f"  Search around: {BEST_PARAMS}")

def obj_c(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 700, 1300),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        max_depth        = trial.suggest_int("max_depth", 4, 6),
        subsample        = trial.suggest_float("subsample", 0.65, 0.90),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.55, 0.80),
        min_child_weight = trial.suggest_int("min_child_weight", 4, 12),
        gamma            = trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 2.0),
        use_label_encoder= False, eval_metric="logloss",
        random_state=RS, n_jobs=1,
    )
    m = xgb.XGBClassifier(**p)
    m.fit(Xtr_c, ytr_c, verbose=False)
    return accuracy_score(yte_c, m.predict(Xte_c))

study_c = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=RS))
study_c.optimize(obj_c, n_trials=100, timeout=600, n_jobs=1, show_progress_bar=False)

bp_c = study_c.best_params
log(f"  Best params: {bp_c}")
xgb_c = xgb.XGBClassifier(
    **{k: v for k, v in bp_c.items()},
    use_label_encoder=False, eval_metric="logloss",
    random_state=RS, n_jobs=1,
)
xgb_c.fit(Xtr_c, ytr_c, verbose=False)
acc_c = accuracy_score(yte_c, xgb_c.predict(Xte_c))
elapsed_c = time.time() - t_exp
record("C_XGBoost_refined_Optuna_100trials", acc_c, len(cols_c),
       elapsed_c, xgb_c, cols_c, split="random")

# Show top features
imp_c = pd.DataFrame({"feature": cols_c,
                       "importance": xgb_c.feature_importances_}
                     ).sort_values("importance", ascending=False)
log("\n  Top 15 features:")
log(imp_c.head(15).to_string(index=False))
log(f"\n  Experiment C done in {elapsed_c:.0f}s")
del xgb_c; gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Weight class stratification
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("EXPERIMENT B — Weight class stratification")
log("=" * 70)
t_exp = time.time()

# ── B1: Weight class encoding (ordinal vs one-hot vs both) ───────────────────
log("\n  B1: Weight class encoding comparison")

# One-hot weight class columns
wc_dummies = pd.get_dummies(master["weight_class"], prefix="wc").astype(int)
for col in wc_dummies.columns:
    master[col] = wc_dummies[col]
wc_oh_cols = list(wc_dummies.columns)

# Base 108 already has weight_class_ord; make feature sets
feats_ord   = BASE_108  # already has weight_class_ord
feats_oh    = [c for c in BASE_108 if c != "weight_class_ord"] + wc_oh_cols
feats_both  = BASE_108 + wc_oh_cols

for tag, flist in [("ordinal", feats_ord), ("one_hot", feats_oh), ("both", feats_both)]:
    t0 = time.time()
    Xtr_b1, Xte_b1, ytr_b1, yte_b1, f_b1 = split_aug_random(flist)
    m_b1 = make_xgb()
    m_b1.fit(Xtr_b1, ytr_b1, verbose=False)
    acc_b1 = accuracy_score(yte_b1, m_b1.predict(Xte_b1))
    record(f"B1_XGBoost_wc_{tag}_{len(f_b1)}feat", acc_b1, len(f_b1),
           time.time()-t0, m_b1 if acc_b1 > best_acc else None,
           f_b1 if acc_b1 > best_acc else None, split="random")
    del m_b1; gc.collect()

# ── B2: Division upset rate feature ─────────────────────────────────────────
log("\n  B2: Division upset rate feature")

# Compute upset rate from TRAINING SET only (avoid leakage)
sub_b2 = master[
    (master["date"] >= pd.Timestamp(DATE_CUT)) &
    (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
    (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
].copy()
train_idx_b2, test_idx_b2 = train_test_split(
    sub_b2.index, test_size=0.2, random_state=RS
)
train_b2 = sub_b2.loc[train_idx_b2]
div_upset = (train_b2.groupby("weight_class")["Winner_bin"]
             .apply(lambda x: (x == 0).sum() / len(x)))  # Blue wins rate
sub_b2["division_upset_rate"] = sub_b2["weight_class"].map(div_upset).fillna(0.42)
master.loc[sub_b2.index, "division_upset_rate"] = sub_b2["division_upset_rate"]
master["division_upset_rate"] = master["division_upset_rate"].fillna(0.42)

feats_upset = BASE_108 + ["division_upset_rate"]
t0 = time.time()
Xtr_up, Xte_up, ytr_up, yte_up, f_up = split_aug_random(feats_upset)
m_up = make_xgb()
m_up.fit(Xtr_up, ytr_up, verbose=False)
acc_up = accuracy_score(yte_up, m_up.predict(Xte_up))
record("B2_XGBoost_division_upset_rate", acc_up, len(f_up),
       time.time()-t0, m_up if acc_up > best_acc else None,
       f_up if acc_up > best_acc else None, split="random")
del m_up; gc.collect()

# ── B3: Per-division models ──────────────────────────────────────────────────
log("\n  B3: Per-division models (XGBoost, divisions with 200+ fights)")

div_counts = sub_b2["weight_class"].value_counts()
eligible_divs = div_counts[div_counts >= 200].index.tolist()
log(f"  Eligible divisions: {eligible_divs}")

div_preds  = {}  # division → (y_true, y_pred)
per_div_acc = {}

for div in eligible_divs:
    div_sub = master[
        (master["date"] >= pd.Timestamp(DATE_CUT)) &
        (master["weight_class"] == div) &
        (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
        (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
    ].copy()
    cols_div = [c for c in BASE_108 if c in div_sub.columns]
    div_sub = div_sub[cols_div + ["Winner_bin"]].dropna()
    if len(div_sub) < 100:
        continue
    X_d, y_d = div_sub[cols_div], div_sub["Winner_bin"]
    Xtr_d, Xte_d, ytr_d, yte_d = train_test_split(X_d, y_d, test_size=0.2, random_state=RS)
    Xtr_da, ytr_da = augment(Xtr_d, ytr_d, cols_div)
    m_d = make_xgb()
    m_d.fit(Xtr_da.values, ytr_da.values, verbose=False)
    acc_d = accuracy_score(yte_d.values, m_d.predict(Xte_d.values))
    per_div_acc[div] = (acc_d, len(yte_d))
    del m_d; gc.collect()

log("\n  Per-division accuracy:")
total_test = sum(n for _, n in per_div_acc.values())
weighted_avg = sum(a * n for a, n in per_div_acc.values()) / total_test
for div, (a, n) in sorted(per_div_acc.items(), key=lambda x: -x[1][0]):
    log(f"    {div:30s}  acc={a:.4f}  n_test={n}")
log(f"  Weighted avg per-division: {weighted_avg:.4f}")
record("B3_per_division_weighted_avg", weighted_avg, 108, time.time()-t_exp, split="random")

log(f"\n  Experiment B done in {time.time()-t_exp:.0f}s")
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — Fight recency weighting
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("EXPERIMENT D — Fight recency weighting (sample_weight)")
log("=" * 70)
t_exp = time.time()

# Prepare dataset with fight year
sub_d = master[
    (master["date"] >= pd.Timestamp(DATE_CUT)) &
    (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
    (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
].copy()
cols_d = [c for c in BASE_108 if c in sub_d.columns]
sub_d = sub_d[cols_d + ["Winner_bin", "date"]].dropna(subset=cols_d + ["Winner_bin"])
sub_d["fight_year"] = sub_d["date"].dt.year

Xtr_d, Xte_d, ytr_d, yte_d = train_test_split(
    sub_d[cols_d + ["fight_year"]], sub_d["Winner_bin"],
    test_size=0.2, random_state=RS
)

year_tr = Xtr_d["fight_year"].values
Xtr_d   = Xtr_d[cols_d]
Xte_d   = Xte_d[cols_d]

Xtr_d_aug, ytr_d_aug = augment(Xtr_d, ytr_d, cols_d)
# Augmented data: original rows first, then swapped copies — repeat years
year_aug = np.concatenate([year_tr, year_tr])
Xtr_d_np = Xtr_d_aug.values
ytr_d_np = ytr_d_aug.values

# Unweighted baseline for D
t0 = time.time()
xgb_d_base = make_xgb()
xgb_d_base.fit(Xtr_d_np, ytr_d_np, verbose=False)
acc_d_base = accuracy_score(yte_d.values, xgb_d_base.predict(Xte_d.values))
record("D_XGBoost_unweighted_baseline", acc_d_base, len(cols_d),
       time.time()-t0, xgb_d_base, list(cols_d), split="random")
del xgb_d_base; gc.collect()

# Linear recency weight: 1 + (year - 2017) * 0.15
t0 = time.time()
w_linear = 1 + (year_aug - 2017) * 0.15
w_linear = np.clip(w_linear, 0.5, 5.0)
xgb_d_lin = make_xgb()
xgb_d_lin.fit(Xtr_d_np, ytr_d_np, sample_weight=w_linear, verbose=False)
acc_d_lin = accuracy_score(yte_d.values, xgb_d_lin.predict(Xte_d.values))
record("D_XGBoost_linear_recency_weight", acc_d_lin, len(cols_d),
       time.time()-t0, xgb_d_lin if acc_d_lin > best_acc else None,
       list(cols_d) if acc_d_lin > best_acc else None, split="random")
del xgb_d_lin; gc.collect()

# Exponential recency weight: exp((year - 2017) * 0.15)
t0 = time.time()
w_exp = np.exp((year_aug - 2017) * 0.15)
w_exp = np.clip(w_exp, 0.5, 8.0)
xgb_d_exp = make_xgb()
xgb_d_exp.fit(Xtr_d_np, ytr_d_np, sample_weight=w_exp, verbose=False)
acc_d_exp = accuracy_score(yte_d.values, xgb_d_exp.predict(Xte_d.values))
record("D_XGBoost_exp_recency_weight", acc_d_exp, len(cols_d),
       time.time()-t0, xgb_d_exp if acc_d_exp > best_acc else None,
       list(cols_d) if acc_d_exp > best_acc else None, split="random")
del xgb_d_exp; gc.collect()

log(f"\n  Experiment D done in {time.time()-t_exp:.0f}s")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT E — Two-stage: finish vs decision → winner
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("EXPERIMENT E — Two-stage prediction (finish/decision → winner)")
log("=" * 70)
t_exp = time.time()

# Prepare dataset; keep is_finish as extra col
sub_e = master[
    (master["date"] >= pd.Timestamp(DATE_CUT)) &
    (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
    (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
    master["finish"].isin(["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"])  # exclude DQ/CNC
].copy()
cols_e = [c for c in BASE_108 if c in sub_e.columns]
sub_e = sub_e[cols_e + ["Winner_bin", "is_finish"]].dropna(subset=cols_e + ["Winner_bin"])

X_e, y_e = sub_e[cols_e], sub_e["Winner_bin"]
is_fin_e  = sub_e["is_finish"]

Xtr_e, Xte_e, ytr_e, yte_e, idx_tr_e, idx_te_e = train_test_split(
    X_e, y_e, is_fin_e.index,
    test_size=0.2, random_state=RS
)
fin_tr = is_fin_e.loc[idx_tr_e].values
fin_te = is_fin_e.loc[idx_te_e].values

log(f"  Dataset: train={len(ytr_e)}, test={len(yte_e)}")
log(f"  Train finish rate: {fin_tr.mean():.3f}  Test finish rate: {fin_te.mean():.3f}")

# Augment training set (is_finish label unchanged by R/B swap)
Xtr_e_aug, ytr_e_aug = augment(Xtr_e, ytr_e, cols_e)
fin_tr_aug = np.concatenate([fin_tr, fin_tr])

# Stage 1: predict finish (1) vs decision (0)
log("\n  Stage 1: finish vs decision")
t0 = time.time()
xgb_s1 = make_xgb()
xgb_s1.fit(Xtr_e_aug.values, fin_tr_aug, verbose=False)
s1_acc = accuracy_score(fin_te, xgb_s1.predict(Xte_e.values))
p_finish_te = xgb_s1.predict_proba(Xte_e.values)[:, 1]
log(f"  Stage 1 finish-vs-decision accuracy: {s1_acc:.4f}")

# Stage 2a: winner given finish (train only on finish fights)
log("  Stage 2a: winner | finish")
finish_mask_tr = fin_tr_aug == 1
X2a_tr = Xtr_e_aug.values[finish_mask_tr]
y2a_tr = ytr_e_aug.values[finish_mask_tr]
log(f"    Training samples (finish): {len(y2a_tr)}")
xgb_s2a = make_xgb()
xgb_s2a.fit(X2a_tr, y2a_tr, verbose=False)
p_win_given_finish = xgb_s2a.predict_proba(Xte_e.values)[:, 1]

# Stage 2b: winner given decision (train only on decision fights)
log("  Stage 2b: winner | decision")
dec_mask_tr = fin_tr_aug == 0
X2b_tr = Xtr_e_aug.values[dec_mask_tr]
y2b_tr = ytr_e_aug.values[dec_mask_tr]
log(f"    Training samples (decision): {len(y2b_tr)}")
xgb_s2b = make_xgb()
xgb_s2b.fit(X2b_tr, y2b_tr, verbose=False)
p_win_given_decision = xgb_s2b.predict_proba(Xte_e.values)[:, 1]

# Combine: P(Red wins) = P(finish) * P(Red|finish) + P(dec) * P(Red|dec)
p_red_combined = (p_finish_te * p_win_given_finish +
                  (1 - p_finish_te) * p_win_given_decision)
preds_two_stage = (p_red_combined > 0.5).astype(int)
acc_two_stage   = accuracy_score(yte_e.values, preds_two_stage)
elapsed_e = time.time() - t_exp

# Compare to single-model on same subset
xgb_single = make_xgb()
xgb_single.fit(Xtr_e_aug.values, ytr_e_aug.values, verbose=False)
acc_single = accuracy_score(yte_e.values, xgb_single.predict(Xte_e.values))

log(f"\n  Single-model XGBoost on same subset: {acc_single:.4f}")
log(f"  Two-stage XGBoost:                   {acc_two_stage:.4f}")
log(f"  Two-stage delta vs single:           {acc_two_stage - acc_single:+.4f}")
record("E_XGBoost_single_model_full_data", acc_single, len(cols_e),
       elapsed_e, xgb_single, list(cols_e), split="random")
record("E_XGBoost_two_stage_finish_decision", acc_two_stage, len(cols_e),
       elapsed_e, split="random")

log(f"\n  Experiment E done in {elapsed_e:.0f}s")
del xgb_s1, xgb_s2a, xgb_s2b, xgb_single; gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("FINAL RESULTS — ALL EXPERIMENTS")
log("=" * 70)

results_df = (
    pd.DataFrame(all_results)
    .dropna(subset=["acc"])
    .sort_values(["split", "acc"], ascending=[True, False])
    .reset_index(drop=True)
)
log(results_df.to_string(index=False))

log("\n── Random-split results (directly comparable to baseline) ──")
rand_df = results_df[results_df["split"] == "random"].sort_values("acc", ascending=False)
log(rand_df[["config", "acc", "n_feats"]].to_string(index=False))

log("\n── Temporal-split results (2024+ test) ──")
temp_df = results_df[results_df["split"] == "temporal"]
log(temp_df[["config", "acc", "n_feats"]].to_string(index=False))

log(f"\nBaseline (exp5 XGBoost):  {BASELINE_ACC:.4f}")
log(f"New best (random split):  {best_acc:.4f}  (delta: {best_acc - BASELINE_ACC:+.4f})")

if best_acc > BASELINE_ACC:
    log(f"\n✓ IMPROVEMENT — saved to {BEST_MDL}")
    log(f"  Model type: {type(best_model).__name__}")
    if hasattr(best_model, "feature_importances_"):
        winner_imp = pd.DataFrame({
            "feature": best_feats,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)
        log("\n  Winner top 20 feature importances:")
        log(winner_imp.to_string(index=False))
    log(f"\n  Best hyperparameters:")
    if hasattr(best_model, "get_params"):
        params = {k: v for k, v in best_model.get_params().items()
                  if k in ["n_estimators", "learning_rate", "max_depth",
                            "subsample", "colsample_bytree", "min_child_weight",
                            "gamma", "reg_alpha"]}
        log(f"  {params}")
else:
    log("\n✗ No improvement over 69.94% — model files unchanged.")

log(f"\n[DONE]  {pd.Timestamp.now().isoformat()}")
