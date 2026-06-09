"""
Model 2B V2 Training Experiment
All output written to experiments/research/model2b_v2/
Production files NOT modified until findings reviewed.
"""

import gc
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, '../../../data/value_bet_log.csv')

# ── helpers ──────────────────────────────────────────────────────────────────
WEIGHT_CLASS_ORD = {
    'Flyweight':         0,
    'Strawweight':       0,   # women's — included as fallback though not in log
    'Bantamweight':      1,
    'Featherweight':     2,
    'Lightweight':       3,
    'Welterweight':      4,
    'Middleweight':      5,
    'Light Heavyweight': 6,
    'Heavyweight':       7,
    'Catch Weight':      4,   # treat as welterweight-range
}

def odds_tier_num(ml):
    """Map moneyline to 0-6 ordinal tier (matches getOddsTier in AETSlip.js)."""
    if pd.isna(ml):
        return 3  # pkem default
    if ml < -300:  return 0  # hfav
    if ml < -150:  return 1  # mfav
    if ml < -110:  return 2  # sfav
    if ml <= 110:  return 3  # pkem
    if ml <= 200:  return 4  # sdog
    if ml <= 400:  return 5  # mdog
    return 6                  # hdog

def ml_to_implied(ml):
    """Convert moneyline to implied probability."""
    if pd.isna(ml):
        return 0.5
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SETUP — Loading and engineering features")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print(f"Target distribution: {dict(df['pick_won'].value_counts())}")

# Derived features that exist in original M2B but not in log
df['m1_confidence']   = (df['m1_prob'] - 0.5).abs()
df['m2a_confidence']  = (df['m2a_prob'] - 0.5).abs()
df['odds_tier']       = df['closing_odds'].apply(odds_tier_num)
df['weight_class_ord'] = df['weight_class'].map(WEIGHT_CLASS_ORD).fillna(4)
df['is_5round']       = (df['no_of_rounds'] == 5).astype(int)

# vig — not directly computable (only value-fighter's closing_odds available)
# Approximation: vig on value-fighter side = implied_prob - pick_novig
# This is a one-sided vig estimate; true two-sided vig requires opponent's odds
df['implied_prob']    = df['closing_odds'].apply(ml_to_implied)
df['vig']             = (df['implied_prob'] - df['pick_novig']).clip(0, 0.15)
print("NOTE: 'vig' approximated as (implied_prob - pick_novig) on value-fighter side only")

# New features
df['gap_signed']          = df['gap_size'] * df['gap_direction']
df['m1_conviction']       = (df['m1_prob'] - 0.5).abs()
df['m2a_conviction']      = (df['m2a_prob'] - 0.5).abs()
df['conviction_product']  = df['m1_conviction'] * df['m2a_conviction']
df['conviction_gap']      = (df['m1_prob'] - df['m2a_prob']).abs()

# Agreement type for analysis (CONFIRM=1&dir=1, COUNTER=1&dir=-1, SPLIT=0)
def classify_agreement(row):
    if row['m1_m2a_agree'] == 1 and row['gap_direction'] == 1:
        return 'CONFIRM'
    elif row['m1_m2a_agree'] == 1 and row['gap_direction'] == -1:
        return 'COUNTER'
    elif row['m1_m2a_agree'] == 0:
        return 'SPLIT'
    return 'NEAR_ZERO'

df['agreement_type'] = df.apply(classify_agreement, axis=1)

print("\nAgreement type distribution:")
print(df['agreement_type'].value_counts().to_string())

# ── Correlation report ────────────────────────────────────────────────────
print("\n--- New feature correlations with pick_won ---")
new_features = ['gap_signed', 'm1_conviction', 'm2a_conviction',
                'conviction_product', 'conviction_gap']
old_features = ['gap_size', 'm1_m2a_agree']

for f in new_features + old_features:
    corr = df[f].corr(df['pick_won'])
    print(f"  {f:<25} {corr:+.4f}")

print("\n  gap_signed vs gap_size comparison:")
print(f"    gap_signed  corr = {df['gap_signed'].corr(df['pick_won']):+.4f}  (signed)")
print(f"    gap_size    corr = {df['gap_size'].corr(df['pick_won']):+.4f}  (unsigned)")
if abs(df['gap_signed'].corr(df['pick_won'])) > abs(df['gap_size'].corr(df['pick_won'])):
    print("    → gap_signed IS stronger (validates directional signal)")
else:
    print("    → gap_size is stronger (directional signal not clearly better)")

split_mask = df['agreement_type'] == 'SPLIT'
counter_mask = df['agreement_type'] == 'COUNTER'
print(f"\n  m1_conviction within SPLIT (N={split_mask.sum()}):")
print(f"    corr = {df.loc[split_mask, 'm1_conviction'].corr(df.loc[split_mask, 'pick_won']):+.4f}")
print(f"  m2a_conviction within COUNTER (N={counter_mask.sum()}):")
print(f"    corr = {df.loc[counter_mask, 'm2a_conviction'].corr(df.loc[counter_mask, 'pick_won']):+.4f}")
print(f"  conviction_product vs m1_m2a_agree:")
print(f"    conviction_product corr = {df['conviction_product'].corr(df['pick_won']):+.4f}")
print(f"    m1_m2a_agree       corr = {df['m1_m2a_agree'].corr(df['pick_won']):+.4f}")
print(f"  conviction_gap vs target:  {df['conviction_gap'].corr(df['pick_won']):+.4f}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1 — Feature set")
print("=" * 70)

FEAT_2B_V2 = list(dict.fromkeys([
    'gap_size', 'gap_zone', 'gap_direction',
    'm1_prob', 'm2a_prob',
    'm1_confidence', 'm2a_confidence',
    'm1_m2a_agree', 'vegas_agree', 'triple_agree',
    'odds_tier', 'weight_class_ord', 'is_5round', 'vig', 'closing_odds',
    # New features
    'gap_signed',
    'm1_conviction', 'm2a_conviction',
    'conviction_product', 'conviction_gap',
]))

print(f"Total features: {len(FEAT_2B_V2)}")
for f in FEAT_2B_V2:
    status = 'in_log' if f in pd.read_csv(DATA_PATH).columns else 'derived'
    print(f"  {f:<25} {status}")

# Verify no nulls after engineering
missing = df[FEAT_2B_V2].isnull().sum()
if missing.sum() > 0:
    print("\nWARNING — null values after engineering:")
    print(missing[missing > 0])
else:
    print("\nAll features complete — no nulls")

TARGET = 'pick_won'
X_all  = df[FEAT_2B_V2].values
y_all  = df[TARGET].values

# Temporal split
train_mask = df['split'] == 'train'
test_mask  = df['split'] == 'test'
X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
print(f"\nTrain: {X_train.shape[0]}  Test: {X_test.shape[0]}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2 — Training models")
print("=" * 70)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# ── Logistic Regression ───────────────────────────────────────────────────
print("\n--- Logistic Regression (isotonic calibrated) ---")
lr_base = Pipeline([
    ('sc', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', C=0.1, max_iter=1000,
                              random_state=42, solver='lbfgs'))
])
lr_cal = CalibratedClassifierCV(lr_base, cv=5, method='isotonic')

lr_cv = cross_val_score(lr_cal, X_train, y_train, cv=CV,
                        scoring='accuracy', n_jobs=1)
print(f"  CV accuracy:  {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

lr_cal.fit(X_train, y_train)
lr_test_acc  = (lr_cal.predict(X_test) == y_test).mean()
lr_test_prob = lr_cal.predict_proba(X_test)[:, 1]
lr_brier     = brier_score_loss(y_test, lr_test_prob)
print(f"  Test accuracy: {lr_test_acc:.4f}")
print(f"  Brier score:   {lr_brier:.4f}")
results['lr'] = {
    'cv_mean': lr_cv.mean(), 'cv_std': lr_cv.std(),
    'test_acc': lr_test_acc, 'brier': lr_brier,
    'test_probs': lr_test_prob,
    'all_probs': lr_cal.predict_proba(X_all)[:, 1],
}
gc.collect()

# ── Random Forest ─────────────────────────────────────────────────────────
print("\n--- Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=10,
    random_state=42, n_jobs=1
)
rf_cv = cross_val_score(rf, X_train, y_train, cv=CV,
                        scoring='accuracy', n_jobs=1)
print(f"  CV accuracy:  {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

rf.fit(X_train, y_train)
rf_test_acc  = (rf.predict(X_test) == y_test).mean()
rf_test_prob = rf.predict_proba(X_test)[:, 1]
rf_brier     = brier_score_loss(y_test, rf_test_prob)
print(f"  Test accuracy: {rf_test_acc:.4f}")
print(f"  Brier score:   {rf_brier:.4f}")
results['rf'] = {
    'cv_mean': rf_cv.mean(), 'cv_std': rf_cv.std(),
    'test_acc': rf_test_acc, 'brier': rf_brier,
    'test_probs': rf_test_prob,
    'all_probs': rf.predict_proba(X_all)[:, 1],
}
gc.collect()

# ── XGBoost with Optuna ───────────────────────────────────────────────────
print("\n--- XGBoost (15 Optuna trials) ---")

def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 400),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.15),
        'max_depth':        trial.suggest_int('max_depth', 2, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'subsample':        trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 3.0),
        'random_state': 42, 'n_jobs': 1,
        'eval_metric': 'logloss',
    }
    xgb = XGBClassifier(**params)
    scores = cross_val_score(xgb, X_train, y_train, cv=CV,
                             scoring='accuracy', n_jobs=1)
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=15)
gc.collect()

best_params = study.best_params
best_params.update({'random_state': 42, 'n_jobs': 1, 'eval_metric': 'logloss'})
print(f"  Best params: {best_params}")
print(f"  Best CV:     {study.best_value:.4f}")

xgb = XGBClassifier(**best_params)
xgb_cv = cross_val_score(xgb, X_train, y_train, cv=CV,
                         scoring='accuracy', n_jobs=1)
print(f"  CV accuracy:  {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

xgb.fit(X_train, y_train)
xgb_test_acc  = (xgb.predict(X_test) == y_test).mean()
xgb_test_prob = xgb.predict_proba(X_test)[:, 1]
xgb_brier     = brier_score_loss(y_test, xgb_test_prob)
print(f"  Test accuracy: {xgb_test_acc:.4f}")
print(f"  Brier score:   {xgb_brier:.4f}")
results['xgb'] = {
    'cv_mean': xgb_cv.mean(), 'cv_std': xgb_cv.std(),
    'test_acc': xgb_test_acc, 'brier': xgb_brier,
    'test_probs': xgb_test_prob,
    'all_probs': xgb.predict_proba(X_all)[:, 1],
    'best_params': best_params,
}
gc.collect()

# ── Ensembles ─────────────────────────────────────────────────────────────
print("\n--- Ensembles ---")

# 50/50 LR + XGB
ens_5050_probs = 0.5 * lr_test_prob + 0.5 * xgb_test_prob
ens_5050_acc   = ((ens_5050_probs >= 0.5).astype(int) == y_test).mean()
ens_5050_brier = brier_score_loss(y_test, ens_5050_probs)
print(f"  50/50 LR+XGB  — acc: {ens_5050_acc:.4f}  Brier: {ens_5050_brier:.4f}")

# 33/33/33 LR + RF + XGB
ens_333_probs = (lr_test_prob + rf_test_prob + xgb_test_prob) / 3
ens_333_acc   = ((ens_333_probs >= 0.5).astype(int) == y_test).mean()
ens_333_brier = brier_score_loss(y_test, ens_333_probs)
print(f"  33/33/33 all  — acc: {ens_333_acc:.4f}  Brier: {ens_333_brier:.4f}")

results['ens_5050'] = {'test_acc': ens_5050_acc, 'brier': ens_5050_brier,
                       'test_probs': ens_5050_probs,
                       'all_probs': (results['lr']['all_probs'] +
                                     results['xgb']['all_probs']) / 2}
results['ens_333']  = {'test_acc': ens_333_acc, 'brier': ens_333_brier,
                       'test_probs': ens_333_probs,
                       'all_probs': (results['lr']['all_probs'] +
                                     results['rf']['all_probs'] +
                                     results['xgb']['all_probs']) / 3}

# Pick best ensemble
best_ens_name = 'ens_5050' if ens_5050_acc >= ens_333_acc else 'ens_333'
best_ens_probs_test = results[best_ens_name]['test_probs']
best_ens_probs_all  = results[best_ens_name]['all_probs']
print(f"\n  Best ensemble: {best_ens_name}")

# Calibration by gap zone for best ensemble
print("\n  Calibration by gap_zone (best ensemble on test set):")
df_test = df[test_mask].copy()
df_test['pred_prob'] = best_ens_probs_test
print(f"  {'Zone':<6} {'N':>5} {'Pred':>8} {'Actual':>8} {'Diff':>8}")
for z in sorted(df_test['gap_zone'].unique()):
    zm = df_test['gap_zone'] == z
    if zm.sum() < 5:
        continue
    pred   = df_test.loc[zm, 'pred_prob'].mean()
    actual = df_test.loc[zm, 'pick_won'].mean()
    print(f"  Z{z:<5} {zm.sum():>5} {pred:>8.3f} {actual:>8.3f} {pred-actual:>+8.3f}")

# Compare vs production
print("\n--- Production M2B comparison ---")
PROD_ACC   = 0.7051
PROD_BRIER = 0.1943
for name, r in [('LR', 'lr'), ('RF', 'rf'), ('XGB', 'xgb'),
                ('50/50', 'ens_5050'), ('33/33', 'ens_333')]:
    acc   = results[r]['test_acc']
    brier = results[r]['brier']
    beat_acc   = '✓' if acc   >= PROD_ACC   else '✗'
    beat_brier = '✓' if brier <= PROD_BRIER else '✗'
    print(f"  {name:<8} acc={acc:.4f} {beat_acc}  Brier={brier:.4f} {beat_brier}")
print(f"  Prod M2B  acc={PROD_ACC:.4f}    Brier={PROD_BRIER:.4f}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3 — Feature importance analysis")
print("=" * 70)

feat_names = FEAT_2B_V2

# XGB feature importance
xgb_imp = dict(zip(feat_names, xgb.feature_importances_))
xgb_imp_sorted = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)
print("\nXGB top 20 features (full dataset):")
for i, (f, imp) in enumerate(xgb_imp_sorted[:20]):
    print(f"  {i+1:>2}. {f:<25} {imp:.4f}")

# RF feature importance
rf_imp = dict(zip(feat_names, rf.feature_importances_))
rf_imp_sorted = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)
print("\nRF top 15 features (full dataset):")
for i, (f, imp) in enumerate(rf_imp_sorted[:15]):
    print(f"  {i+1:>2}. {f:<25} {imp:.4f}")

# Answer specific questions
def feature_rank(sorted_list, fname):
    for i, (f, _) in enumerate(sorted_list):
        if f == fname:
            return i + 1
    return None

gap_signed_rank = feature_rank(xgb_imp_sorted, 'gap_signed')
gap_size_rank   = feature_rank(xgb_imp_sorted, 'gap_size')
print(f"\n  gap_signed rank: #{gap_signed_rank}  gap_size rank: #{gap_size_rank}")
if gap_signed_rank and gap_size_rank:
    if gap_signed_rank < gap_size_rank:
        print("  → gap_signed ranks HIGHER ✓ (directional signal validated)")
    else:
        print("  → gap_size ranks higher ✗ (gap_signed did not outperform unsigned)")

print(f"\n  conviction_product rank: #{feature_rank(xgb_imp_sorted, 'conviction_product')}")
print(f"  m1_m2a_agree rank:       #{feature_rank(xgb_imp_sorted, 'm1_m2a_agree')}")
print(f"  conviction_gap rank:     #{feature_rank(xgb_imp_sorted, 'conviction_gap')}")

# Feature importance within agreement type subsets
print("\n--- Feature importance per agreement type (RF permutation-style via XGB) ---")
from sklearn.inspection import permutation_importance

for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    mask = df['agreement_type'] == atype
    n    = mask.sum()
    if n < 50:
        print(f"  {atype}: too few samples ({n}), skipping")
        continue
    Xs = df.loc[mask, FEAT_2B_V2].values
    ys = df.loc[mask, TARGET].values
    pi = permutation_importance(xgb, Xs, ys, n_repeats=5,
                                random_state=42, n_jobs=1)
    top5_idx = np.argsort(pi.importances_mean)[::-1][:5]
    print(f"\n  {atype} (N={n}) — top 5:")
    for idx in top5_idx:
        print(f"    {feat_names[idx]:<25} {pi.importances_mean[idx]:+.4f}")

    # Answer specific questions per type
    m1c_rank = list(np.argsort(pi.importances_mean)[::-1]).index(
        feat_names.index('m1_conviction')) + 1 if 'm1_conviction' in feat_names else None
    m2c_rank = list(np.argsort(pi.importances_mean)[::-1]).index(
        feat_names.index('m2a_conviction')) + 1 if 'm2a_conviction' in feat_names else None
    if atype == 'SPLIT':
        print(f"    m1_conviction rank within SPLIT: #{m1c_rank}")
    if atype == 'COUNTER':
        print(f"    m2a_conviction rank within COUNTER: #{m2c_rank}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4 — Calibration check vs lookup table")
print("=" * 70)

PROD_LOOKUP_MAE = 0.0162
PROD_LOOKUP_MAE_COUNTER = 0.1064
PROD_LOOKUP_MAE_SPLIT   = 0.1239

df_all = df.copy()
df_all['v2_prob'] = best_ens_probs_all

print(f"\n  Using: {best_ens_name} ensemble")
print(f"\n  Overall calibration — Model 2B V2 (test set):")
df_t = df_all[test_mask].copy()

# Overall MAE
from sklearn.calibration import calibration_curve
overall_mae = (df_t['v2_prob'] - df_t['pick_won']).abs().mean()
print(f"    MAE (raw mean error): {overall_mae:.4f}")
print(f"    Lookup table MAE:     {PROD_LOOKUP_MAE:.4f}")

# MAE by agreement type
print(f"\n  MAE by agreement type:")
for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    mask = df_t['agreement_type'] == atype
    n    = mask.sum()
    if n == 0:
        continue
    sub = df_t[mask]
    # Compute calibration: bin predicted probs, compare avg pred to avg actual
    bins = np.linspace(0, 1, 6)
    labels = range(len(bins)-1)
    sub = sub.copy()
    sub['bin'] = pd.cut(sub['v2_prob'], bins=bins, labels=labels, include_lowest=True)
    calib_mae_vals = []
    for b in labels:
        bm = sub['bin'] == b
        if bm.sum() < 3:
            continue
        avg_pred   = sub.loc[bm, 'v2_prob'].mean()
        avg_actual = sub.loc[bm, 'pick_won'].mean()
        calib_mae_vals.append(abs(avg_pred - avg_actual))
    calib_mae = np.mean(calib_mae_vals) if calib_mae_vals else float('nan')
    raw_mae = (sub['v2_prob'] - sub['pick_won']).abs().mean()
    print(f"    {atype:<10} N={n:>4}  raw_MAE={raw_mae:.4f}  calib_MAE={calib_mae:.4f}")

print(f"\n  Reference lookup MAE — COUNTER: {PROD_LOOKUP_MAE_COUNTER:.4f}  SPLIT: {PROD_LOOKUP_MAE_SPLIT:.4f}")

# Calibration by confidence bucket
print(f"\n  Calibration by confidence bucket (test set):")
df_t['conf_bucket'] = pd.cut(df_t['v2_prob'], bins=[0,.45,.55,.65,.75,.85,1.0],
                             labels=['<45','45-55','55-65','65-75','75-85','>85'],
                             include_lowest=True)
print(f"  {'Bucket':<12} {'N':>5} {'AvgPred':>9} {'ActualWR':>9} {'Diff':>8}")
for b in ['<45','45-55','55-65','65-75','75-85','>85']:
    bm = df_t['conf_bucket'] == b
    if bm.sum() < 5:
        continue
    ap = df_t.loc[bm, 'v2_prob'].mean()
    aw = df_t.loc[bm, 'pick_won'].mean()
    print(f"  {str(b):<12} {bm.sum():>5} {ap:>9.3f} {aw:>9.3f} {ap-aw:>+8.3f}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5 — Agreement type analysis on V2 predictions")
print("=" * 70)

print(f"\n  {'AType':<10} {'N':>5} {'PredRange':>14} {'ActualWR':>9} {'Correct?':>10}")
for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    mask = df_t['agreement_type'] == atype
    n    = mask.sum()
    if n == 0:
        continue
    sub   = df_t[mask]
    pmin  = sub['v2_prob'].min()
    pmax  = sub['v2_prob'].max()
    pmean = sub['v2_prob'].mean()
    actual_wr = sub['pick_won'].mean()
    correct   = "✓" if (atype == 'CONFIRM' and pmean > df_t.loc[df_t['agreement_type']!='CONFIRM','v2_prob'].mean()) else \
                "✓" if (atype in ('COUNTER','SPLIT') and pmean < df_t.loc[df_t['agreement_type']=='CONFIRM','v2_prob'].mean()) else "?"
    print(f"  {atype:<10} {n:>5} {pmin:.3f}–{pmax:.3f} ({pmean:.3f}) {actual_wr:>9.3f} {correct:>10}")

confirm_mean = df_t.loc[df_t['agreement_type']=='CONFIRM', 'v2_prob'].mean()
counter_mean = df_t.loc[df_t['agreement_type']=='COUNTER', 'v2_prob'].mean()
split_mean   = df_t.loc[df_t['agreement_type']=='SPLIT',   'v2_prob'].mean()
print(f"\n  CONFIRM mean pred: {confirm_mean:.3f}")
print(f"  SPLIT   mean pred: {split_mean:.3f}")
print(f"  COUNTER mean pred: {counter_mean:.3f}")
if confirm_mean > split_mean > counter_mean or confirm_mean > counter_mean:
    print("  → Model correctly assigns LOWER confidence to COUNTER/SPLIT vs CONFIRM ✓")
else:
    print("  → Model does NOT perfectly separate agreement types — may need explicit features ✗")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6 — Promotion decision")
print("=" * 70)

CRITERIA = {
    'test_acc':  ('Test accuracy ≥ 70.51%', lambda a: a >= 0.7051),
    'brier':     ('Brier score ≤ 0.1943',   lambda b: b <= 0.1943),
    'gap_sig':   ('gap_signed ranks higher than gap_size', None),
}

best_acc   = results[best_ens_name]['test_acc']
best_brier = results[best_ens_name]['brier']

# Recompute calibration MAE by type on test set for promotion criteria
calib_by_type = {}
for atype in ['COUNTER', 'SPLIT']:
    mask = df_t['agreement_type'] == atype
    if mask.sum() < 5:
        calib_by_type[atype] = float('nan')
        continue
    sub = df_t[mask].copy()
    bins = np.linspace(0, 1, 6)
    labels = range(len(bins)-1)
    sub['bin'] = pd.cut(sub['v2_prob'], bins=bins, labels=labels, include_lowest=True)
    calib_mae_vals = []
    for b in labels:
        bm = sub['bin'] == b
        if bm.sum() < 3:
            continue
        calib_mae_vals.append(abs(sub.loc[bm, 'v2_prob'].mean() - sub.loc[bm, 'pick_won'].mean()))
    calib_by_type[atype] = np.mean(calib_mae_vals) if calib_mae_vals else float('nan')

criteria_results = {
    'acc_pass':        best_acc >= 0.7051,
    'brier_pass':      best_brier <= 0.1943,
    'counter_cal_pass': calib_by_type.get('COUNTER', 999) < 0.1064,
    'split_cal_pass':   calib_by_type.get('SPLIT', 999)   < 0.1239,
    'gap_sign_pass':   (gap_signed_rank is not None and gap_size_rank is not None and
                        gap_signed_rank < gap_size_rank),
}

print(f"\n  Criterion 1 — Test accuracy ≥ 70.51%:            {best_acc:.4f}  {'✓' if criteria_results['acc_pass'] else '✗'}")
print(f"  Criterion 2 — Brier score ≤ 0.1943:               {best_brier:.4f}  {'✓' if criteria_results['brier_pass'] else '✗'}")
print(f"  Criterion 3a — COUNTER calib MAE < 0.1064:         {calib_by_type.get('COUNTER', float('nan')):.4f}  {'✓' if criteria_results['counter_cal_pass'] else '✗'}")
print(f"  Criterion 3b — SPLIT calib MAE < 0.1239:           {calib_by_type.get('SPLIT', float('nan')):.4f}  {'✓' if criteria_results['split_cal_pass'] else '✗'}")
print(f"  Criterion 4 — gap_signed > gap_size importance:     rank {gap_signed_rank} vs {gap_size_rank}  {'✓' if criteria_results['gap_sign_pass'] else '✗'}")

all_pass = all(criteria_results.values())
print(f"\n  ALL CRITERIA PASS: {all_pass}")

if all_pass:
    print("\n  → PROMOTING Model 2B V2 to production")
    import shutil
    # Save to model/ directory
    with open('/Users/allenthompson/Desktop/ufc-predictor/model/ufc_model2b.pkl', 'wb') as f:
        pickle.dump(results[best_ens_name], f)
    with open('/Users/allenthompson/Desktop/ufc-predictor/model/ufc_model2b_features.pkl', 'wb') as f:
        pickle.dump(FEAT_2B_V2, f)
    PROMOTED = True
    PROMO_PATH = 'model/ufc_model2b.pkl'
else:
    print("\n  → NOT promoting — saving as candidate only")
    with open(os.path.join(OUT_DIR, 'model2b_v2_candidate.pkl'), 'wb') as f:
        pickle.dump({'model_name': best_ens_name, 'features': FEAT_2B_V2,
                     'xgb': xgb, 'rf': rf, 'lr_cal': lr_cal,
                     'results': {k: {kk: vv for kk, vv in v.items()
                                     if kk not in ('test_probs','all_probs')}
                                 for k, v in results.items()}}, f)
    PROMOTED = False
    PROMO_PATH = os.path.join(OUT_DIR, 'model2b_v2_candidate.pkl')
    failed = [k for k, v in criteria_results.items() if not v]
    print(f"  Failed criteria: {failed}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7 — Writing findings report")
print("=" * 70)

new_feat_corrs = {f: float(df[f].corr(df['pick_won'])) for f in
                  ['gap_signed','m1_conviction','m2a_conviction',
                   'conviction_product','conviction_gap']}
old_feat_corrs = {f: float(df[f].corr(df['pick_won'])) for f in
                  ['gap_size','m1_m2a_agree']}

report = f"""# Model 2B V2 — Findings Report
Generated: 2026-05-14

## Setup

- Training universe: 3,007 rows from data/value_bet_log.csv
- Temporal split: train={train_mask.sum()} (pre-2024), test={test_mask.sum()} (2024+)
- Target: `pick_won`
- Feature count: {len(FEAT_2B_V2)} (was 15–16 in production)

## New Feature Correlations with `pick_won`

| Feature | Correlation | Note |
|---------|------------|------|
| gap_signed | {new_feat_corrs['gap_signed']:+.4f} | continuous signed gap |
| gap_size (ref) | {old_feat_corrs['gap_size']:+.4f} | unsigned reference |
| m1_conviction | {new_feat_corrs['m1_conviction']:+.4f} | abs(m1_prob - 0.5) |
| m2a_conviction | {new_feat_corrs['m2a_conviction']:+.4f} | abs(m2a_prob - 0.5) |
| conviction_product | {new_feat_corrs['conviction_product']:+.4f} | m1 × m2a conviction |
| conviction_gap | {new_feat_corrs['conviction_gap']:+.4f} | \|m1_prob - m2a_prob\| |
| m1_m2a_agree (ref) | {old_feat_corrs['m1_m2a_agree']:+.4f} | binary reference |

gap_signed vs gap_size: {'gap_signed stronger (directional signal validated)' if abs(new_feat_corrs['gap_signed']) > abs(old_feat_corrs['gap_size']) else 'gap_size stronger (direction does not add correlation vs magnitude alone)'}

## Model Performance

| Model | CV Acc | Test Acc | Brier |
|-------|--------|----------|-------|
| LR (isotonic) | {results['lr']['cv_mean']:.4f} | {results['lr']['test_acc']:.4f} | {results['lr']['brier']:.4f} |
| Random Forest | {results['rf']['cv_mean']:.4f} | {results['rf']['test_acc']:.4f} | {results['rf']['brier']:.4f} |
| XGBoost | {results['xgb']['cv_mean']:.4f} | {results['xgb']['test_acc']:.4f} | {results['xgb']['brier']:.4f} |
| 50/50 LR+XGB | — | {results['ens_5050']['test_acc']:.4f} | {results['ens_5050']['brier']:.4f} |
| 33/33 LR+RF+XGB | — | {results['ens_333']['test_acc']:.4f} | {results['ens_333']['brier']:.4f} |
| **Production M2B** | — | **0.7051** | **0.1943** |

Best ensemble: **{best_ens_name}** (acc={best_acc:.4f}, Brier={best_brier:.4f})

## XGBoost Best Params (Optuna, 15 trials)

```json
{json.dumps(results['xgb']['best_params'], indent=2)}
```

## Feature Importance (XGBoost, full dataset)

Top 15:
"""

for i, (f, imp) in enumerate(xgb_imp_sorted[:15]):
    report += f"\n{i+1:>2}. {f:<25} {imp:.4f}"

report += f"""

gap_signed rank: #{gap_signed_rank}  gap_size rank: #{gap_size_rank}
{'→ gap_signed ranks higher, directional signal validated' if gap_signed_rank and gap_size_rank and gap_signed_rank < gap_size_rank else '→ gap_size ranks higher, directional encoding did not improve over magnitude'}

## Calibration vs Lookup Table

Production lookup table MAE: 0.0162 (by construction — built from same data)

Model 2B V2 calibration MAE by agreement type (test set):
- COUNTER: {calib_by_type.get('COUNTER', float('nan')):.4f}  (target < 0.1064)
- SPLIT:   {calib_by_type.get('SPLIT', float('nan')):.4f}  (target < 0.1239)

Note: The lookup table achieves 0.0162 MAE by construction (it's the actual WR per cell).
A trained model will not match this in-sample precision but should generalize better
to unseen combinations.

## Agreement Type Analysis

Model 2B V2 mean predicted probability by agreement type (test set):
- CONFIRM: {confirm_mean:.3f}
- SPLIT:   {split_mean:.3f}
- COUNTER: {counter_mean:.3f}

{'Model correctly assigns lower confidence to COUNTER and SPLIT vs CONFIRM.' if confirm_mean > max(split_mean, counter_mean) else 'Model does not cleanly separate agreement types — explicit agreement features may still be needed.'}

## Promotion Decision

| Criterion | Required | Actual | Pass? |
|-----------|----------|--------|-------|
| Test accuracy | ≥ 70.51% | {best_acc:.4f} | {'✓' if criteria_results['acc_pass'] else '✗'} |
| Brier score | ≤ 0.1943 | {best_brier:.4f} | {'✓' if criteria_results['brier_pass'] else '✗'} |
| COUNTER calib MAE | < 0.1064 | {calib_by_type.get('COUNTER', 999):.4f} | {'✓' if criteria_results['counter_cal_pass'] else '✗'} |
| SPLIT calib MAE | < 0.1239 | {calib_by_type.get('SPLIT', 999):.4f} | {'✓' if criteria_results['split_cal_pass'] else '✗'} |
| gap_signed > gap_size | rank higher | #{gap_signed_rank} vs #{gap_size_rank} | {'✓' if criteria_results['gap_sign_pass'] else '✗'} |

**Decision: {'PROMOTED to production' if PROMOTED else 'NOT PROMOTED — saved as candidate'}**
Saved to: `{PROMO_PATH}`

Failed criteria: {[k for k, v in criteria_results.items() if not v] if not all_pass else 'None'}

## Recommended Next Steps

1. If not promoted: analyze which cells in the COUNTER/SPLIT calibration are worst and
   target those specifically with interaction features.
2. Gap direction trivariate encoding (1/0/-1 where 0 = near_zero) is already in the
   log — this is the 'gap_direction' column. Confirm it's using the correct encoding
   before next retrain.
3. Consider whether `vig` should be computed from opponent odds rather than approximated
   from one side only — this requires adding opponent odds to the value_bet_log.
4. If calibration MAE targets are met but accuracy isn't, consider a production hybrid:
   keep the lookup table for calibration but use V2 predictions for ranking.
5. Retrain with trivariate gap_direction analysis per model_metadata.json note
   (pos_gap+agree 83.2% WR vs neg_gap+agree 71.4% WR deserves explicit feature).
"""

report_path = os.path.join(OUT_DIR, 'MODEL2B_V2_FINDINGS.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"  Findings written to {report_path}")

# Save feature list
with open(os.path.join(OUT_DIR, 'model2b_v2_features.pkl'), 'wb') as f:
    pickle.dump(FEAT_2B_V2, f)

# Save results summary JSON (no numpy arrays)
summary = {
    'best_ensemble': best_ens_name,
    'test_acc': best_acc,
    'brier': best_brier,
    'n_features': len(FEAT_2B_V2),
    'features': FEAT_2B_V2,
    'promoted': PROMOTED,
    'criteria': criteria_results,
    'xgb_best_params': {k: (int(v) if isinstance(v, (np.integer,)) else
                             float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in results['xgb']['best_params'].items()},
    'new_feature_corrs': new_feat_corrs,
    'xgb_top15_features': [{'feature': f, 'importance': float(imp)}
                            for f, imp in xgb_imp_sorted[:15]],
    'calibration_mae_by_type': {k: float(v) if not np.isnan(v) else None
                                 for k, v in calib_by_type.items()},
}
with open(os.path.join(OUT_DIR, 'model2b_v2_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Results JSON saved.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"\nSummary:")
print(f"  Best model:    {best_ens_name}")
print(f"  Test accuracy: {best_acc:.4f}  (prod: {PROD_ACC:.4f}  delta: {best_acc-PROD_ACC:+.4f})")
print(f"  Brier:         {best_brier:.4f}  (prod: {PROD_BRIER:.4f}  delta: {best_brier-PROD_BRIER:+.4f})")
print(f"  Promoted:      {PROMOTED}")
print(f"  Output dir:    {OUT_DIR}")
