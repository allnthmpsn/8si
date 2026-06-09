"""
save_xgb_temporal.py — Train XGBoost on temporal split and save for backend blend.

Uses best params from experiment 6, trained on 2017-2023 temporal split.
Saves to model/ufc_model_xgb.pkl + model/feature_columns_xgb.pkl
"""
import warnings
warnings.filterwarnings("ignore")

import joblib, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

RS             = 42
DATE_CUT       = "2017-01-01"
MIN_UFC_FIGHTS = 3
TEMPORAL_CUT   = "2024-01-01"

# Best XGB params from experiment 6
XGB_PARAMS = {
    'n_estimators':    1161,
    'learning_rate':   0.04923008174328196,
    'max_depth':       5,
    'subsample':       0.7723583130801059,
    'colsample_bytree': 0.7646633972324445,
    'min_child_weight': 4,
    'gamma':           0.7458497935545471,
    'reg_alpha':       1.1286735251127924,
    'use_label_encoder': False,
    'eval_metric':     'logloss',
    'random_state':    RS,
    'n_jobs':          -1,
}

print("Loading data and engineering features...")

career_raw = pd.read_csv("data/career_fights.csv")
career_raw["date"] = pd.to_datetime(career_raw["date"])
master     = pd.read_csv("data/ufc-master.csv", low_memory=False)
master["date"] = pd.to_datetime(master["date"])
fighters   = pd.read_csv("data/ufc_fighters_final.csv")

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

# Load the 108 feature list
feat_list = joblib.load("model/feature_columns_best.pkl")
cols = [c for c in feat_list if c in master.columns]
print(f"Available features: {len(cols)} / {len(feat_list)}")

# Build temporal split
sub = master[
    (master["date"] >= pd.Timestamp(DATE_CUT)) &
    (master["R_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS) &
    (master["B_cum_fights"].fillna(0) >= MIN_UFC_FIGHTS)
].copy()
sub = sub[cols + ["Winner_bin", "date"]].dropna(subset=cols + ["Winner_bin"])

train = sub[sub["date"] <  pd.Timestamp(TEMPORAL_CUT)]
test  = sub[sub["date"] >= pd.Timestamp(TEMPORAL_CUT)]
Xtr, ytr = train[cols], train["Winner_bin"]
Xte, yte = test[cols].values, test["Winner_bin"].values

# Augment training set
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

Xtr_a, ytr_a = augment(Xtr, ytr, cols)
print(f"Train: {len(Xtr_a)} (augmented), Test: {len(Xte)}")

# Train XGBoost
print("Training XGBoost on temporal split...")
model_xgb = xgb.XGBClassifier(**XGB_PARAMS)
model_xgb.fit(Xtr_a.values, ytr_a.values, verbose=False)

xgb_acc = accuracy_score(yte, model_xgb.predict(Xte))
print(f"XGB temporal accuracy: {xgb_acc:.4f}")

# Verify blend accuracy
lr_model = joblib.load("model/ufc_model_best.pkl")
lr_proba  = lr_model.predict_proba(Xte)
xgb_proba = model_xgb.predict_proba(Xte)

for lr_w in [0.7, 0.8, 0.9]:
    xgb_w = 1.0 - lr_w
    blend = lr_w * lr_proba + xgb_w * xgb_proba
    acc = accuracy_score(yte, blend.argmax(axis=1))
    print(f"  LR{lr_w:.0%} + XGB{xgb_w:.0%}: {acc:.4f}")

# Save
joblib.dump(model_xgb, "model/ufc_model_xgb.pkl")
joblib.dump(cols, "model/feature_columns_xgb.pkl")
print(f"\nSaved: model/ufc_model_xgb.pkl ({len(cols)} features)")
print("Done.")
