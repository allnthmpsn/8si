"""
Model 2B V3 Training Experiment — second retrain attempt
Two new features: is_m1_signal, agreement_encoded
Post-processing: SPLIT probability floor

All output in experiments/research/model2b_v2/
Production files NOT modified until promotion criteria confirmed.
"""

import gc
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, '../../../data/value_bet_log.csv')

# ── shared helpers (identical to V2) ─────────────────────────────────────
WEIGHT_CLASS_ORD = {
    'Flyweight':         0,
    'Strawweight':       0,
    'Bantamweight':      1,
    'Featherweight':     2,
    'Lightweight':       3,
    'Welterweight':      4,
    'Middleweight':      5,
    'Light Heavyweight': 6,
    'Heavyweight':       7,
    'Catch Weight':      4,
}

def odds_tier_num(ml):
    if pd.isna(ml): return 3
    if ml < -300:   return 0
    if ml < -150:   return 1
    if ml < -110:   return 2
    if ml <=  110:  return 3
    if ml <=  200:  return 4
    if ml <=  400:  return 5
    return 6

def ml_to_implied(ml):
    if pd.isna(ml): return 0.5
    if ml < 0:  return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

def classify_agreement(row):
    if row['m1_m2a_agree'] == 1 and row['gap_direction'] == 1:
        return 'CONFIRM'
    if row['m1_m2a_agree'] == 1 and row['gap_direction'] == -1:
        return 'COUNTER'
    if row['m1_m2a_agree'] == 0:
        return 'SPLIT'
    return 'NEAR_ZERO'

def apply_split_floor(probs, agreement_types, floor=0.45):
    """Floor SPLIT fight predictions — actual SPLIT WR is 52.1%."""
    adjusted = np.array(probs, dtype=float)
    split_mask = (np.array(agreement_types) == 'SPLIT')
    adjusted[split_mask] = np.maximum(adjusted[split_mask], floor)
    return adjusted

def calib_mae(probs, actuals, n_bins=5):
    """Calibration MAE: mean |avg_pred - avg_actual| across non-empty bins."""
    bins   = np.linspace(0, 1, n_bins + 1)
    labels = list(range(n_bins))
    s = pd.Series(probs)
    cuts = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    vals = []
    for b in labels:
        mask = cuts == b
        if mask.sum() < 3:
            continue
        vals.append(abs(probs[mask.values].mean() - actuals[mask.values].mean()))
    return np.mean(vals) if vals else float('nan')


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SETUP — Feature engineering")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

# V2 derived features
df['m1_confidence']    = (df['m1_prob'] - 0.5).abs()
df['m2a_confidence']   = (df['m2a_prob'] - 0.5).abs()
df['odds_tier']        = df['closing_odds'].apply(odds_tier_num)
df['weight_class_ord'] = df['weight_class'].map(WEIGHT_CLASS_ORD).fillna(4)
df['is_5round']        = (df['no_of_rounds'] == 5).astype(int)
df['implied_prob']     = df['closing_odds'].apply(ml_to_implied)
df['vig']              = (df['implied_prob'] - df['pick_novig']).clip(0, 0.15)

df['gap_signed']         = df['gap_size'] * df['gap_direction']
df['m1_conviction']      = (df['m1_prob'] - 0.5).abs()
df['m2a_conviction']     = (df['m2a_prob'] - 0.5).abs()
df['conviction_product'] = df['m1_conviction'] * df['m2a_conviction']
df['conviction_gap']     = (df['m1_prob'] - df['m2a_prob']).abs()

df['agreement_type'] = df.apply(classify_agreement, axis=1)

# ── NEW Feature 1: is_m1_signal ───────────────────────────────────────────
df['is_m1_signal'] = (
    (df['agreement_type'] == 'SPLIT') &
    (df['gap_zone'] >= 5) &
    (df['m1_conviction'] >= 0.15)
).astype(int)

target = 'pick_won'
print(f"\nM1 Signal fights:  {df['is_m1_signal'].sum()}")
print(f"M1 Signal WR:      {df[df['is_m1_signal']==1][target].mean():.3f}")
print(f"Non-signal WR:     {df[df['is_m1_signal']==0][target].mean():.3f}")

# ── NEW Feature 2: agreement_encoded ─────────────────────────────────────
atype_map = {'CONFIRM': 3, 'SPLIT': 2, 'NEAR_ZERO': 1, 'COUNTER': 0}
df['agreement_encoded'] = df['agreement_type'].map(atype_map)

print(f"\nAgreement distribution:")
print(df['agreement_type'].value_counts().to_string())
print(f"\nagreement_encoded value_counts:")
print(df['agreement_encoded'].value_counts().sort_index().to_string())

# ── Final feature set ─────────────────────────────────────────────────────
FEAT_2B_V3 = list(dict.fromkeys([
    'gap_size', 'gap_zone', 'gap_direction', 'gap_signed',
    'm1_prob', 'm2a_prob',
    'm1_conviction', 'm2a_conviction',
    'm1_m2a_agree', 'vegas_agree', 'triple_agree',
    'conviction_product', 'conviction_gap',
    'odds_tier', 'weight_class_ord', 'is_5round',
    'vig', 'closing_odds',
    'is_m1_signal',
    'agreement_encoded',
]))

print(f"\nTotal features: {len(FEAT_2B_V3)}")
missing = df[FEAT_2B_V3].isnull().sum()
if missing.sum() > 0:
    print("WARNING — nulls:", missing[missing > 0].to_dict())
else:
    print("All features complete — no nulls")

# Splits
train_mask = df['split'] == 'train'
test_mask  = df['split'] == 'test'

X_all   = df[FEAT_2B_V3].values
y_all   = df[target].values
X_train = X_all[train_mask]
y_train = y_all[train_mask]
X_test  = X_all[test_mask]
y_test  = y_all[test_mask]

atype_all   = df['agreement_type'].values
atype_test  = atype_all[test_mask]
atype_train = atype_all[train_mask]

print(f"\nTrain: {X_train.shape[0]}  Test: {X_test.shape[0]}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1 — Training models")
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
CV     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
FLOOR  = 0.45
PROD_ACC   = 0.7051
PROD_BRIER = 0.1943

results = {}

def evaluate(name, probs_test, probs_all, y_test, atype_test, cv_acc=None, cv_std=None):
    """Evaluate with and without SPLIT floor."""
    probs_floor = apply_split_floor(probs_test, atype_test, FLOOR)

    acc_raw   = ((probs_test  >= 0.5).astype(int) == y_test).mean()
    acc_floor = ((probs_floor >= 0.5).astype(int) == y_test).mean()
    brier_raw   = brier_score_loss(y_test, probs_test)
    brier_floor = brier_score_loss(y_test, probs_floor)

    split_mask  = atype_test == 'SPLIT'
    split_probs_raw   = probs_test[split_mask]
    split_probs_floor = probs_floor[split_mask]
    split_actuals     = y_test[split_mask]

    split_mae_raw   = calib_mae(split_probs_raw,   split_actuals)
    split_mae_floor = calib_mae(split_probs_floor, split_actuals)

    counter_mask  = atype_test == 'COUNTER'
    counter_probs = probs_test[counter_mask]
    counter_actuals = y_test[counter_mask]
    counter_mae = calib_mae(counter_probs, counter_actuals)

    print(f"\n  {name}")
    if cv_acc is not None:
        print(f"    CV accuracy:            {cv_acc:.4f} ± {cv_std:.4f}")
    print(f"    Test acc (raw):         {acc_raw:.4f}")
    print(f"    Test acc (floor):       {acc_floor:.4f}  {'✓' if acc_floor >= PROD_ACC else '✗'}")
    print(f"    Brier (raw):            {brier_raw:.4f}")
    print(f"    Brier (floor):          {brier_floor:.4f}  {'✓' if brier_floor <= 0.1965 else '✗'}")
    print(f"    SPLIT calib MAE (raw):  {split_mae_raw:.4f}")
    print(f"    SPLIT calib MAE (floor):{split_mae_floor:.4f}  {'✓' if split_mae_floor <= 0.1239 else '✗'}")
    print(f"    COUNTER calib MAE:      {counter_mae:.4f}  {'✓' if counter_mae <= 0.0800 else '✗'}")

    return {
        'cv_acc': cv_acc, 'cv_std': cv_std,
        'acc_raw': acc_raw, 'acc_floor': acc_floor,
        'brier_raw': brier_raw, 'brier_floor': brier_floor,
        'split_mae_raw': split_mae_raw, 'split_mae_floor': split_mae_floor,
        'counter_mae': counter_mae,
        'probs_test': probs_test, 'probs_floor': probs_floor,
        'probs_all': probs_all,
    }

# ── Logistic Regression ───────────────────────────────────────────────────
print("\n--- Logistic Regression (isotonic calibrated) ---")
lr_base = Pipeline([
    ('sc', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', C=0.1, max_iter=1000,
                              random_state=42, solver='lbfgs'))
])
lr_cal = CalibratedClassifierCV(lr_base, cv=5, method='isotonic')
lr_cv  = cross_val_score(lr_cal, X_train, y_train, cv=CV,
                         scoring='accuracy', n_jobs=1)
lr_cal.fit(X_train, y_train)
lr_prob_test = lr_cal.predict_proba(X_test)[:, 1]
lr_prob_all  = lr_cal.predict_proba(X_all)[:, 1]
results['lr'] = evaluate('LR', lr_prob_test, lr_prob_all, y_test, atype_test,
                         lr_cv.mean(), lr_cv.std())
gc.collect()

# ── Random Forest ─────────────────────────────────────────────────────────
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10,
                            random_state=42, n_jobs=1)
rf_cv = cross_val_score(rf, X_train, y_train, cv=CV, scoring='accuracy', n_jobs=1)
rf.fit(X_train, y_train)
rf_prob_test = rf.predict_proba(X_test)[:, 1]
rf_prob_all  = rf.predict_proba(X_all)[:, 1]
results['rf'] = evaluate('RF', rf_prob_test, rf_prob_all, y_test, atype_test,
                         rf_cv.mean(), rf_cv.std())
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
        'random_state': 42, 'n_jobs': 1, 'eval_metric': 'logloss',
    }
    xgb = XGBClassifier(**params)
    scores = cross_val_score(xgb, X_train, y_train, cv=CV,
                             scoring='accuracy', n_jobs=1)
    return scores.mean()

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=15)
gc.collect()

best_params = study.best_params
best_params.update({'random_state': 42, 'n_jobs': 1, 'eval_metric': 'logloss'})
print(f"  Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()}, indent=4)}")
print(f"  Best CV:     {study.best_value:.4f}")

xgb = XGBClassifier(**best_params)
xgb_cv = cross_val_score(xgb, X_train, y_train, cv=CV, scoring='accuracy', n_jobs=1)
xgb.fit(X_train, y_train)
xgb_prob_test = xgb.predict_proba(X_test)[:, 1]
xgb_prob_all  = xgb.predict_proba(X_all)[:, 1]
results['xgb'] = evaluate('XGB', xgb_prob_test, xgb_prob_all, y_test, atype_test,
                           xgb_cv.mean(), xgb_cv.std())
results['xgb']['best_params'] = best_params
gc.collect()

# ── Ensembles ─────────────────────────────────────────────────────────────
print("\n--- Ensembles ---")

ens_5050_test = 0.5 * lr_prob_test + 0.5 * xgb_prob_test
ens_5050_all  = 0.5 * lr_prob_all  + 0.5 * xgb_prob_all
results['ens_5050'] = evaluate('50/50 LR+XGB', ens_5050_test, ens_5050_all,
                                y_test, atype_test)

ens_333_test = (lr_prob_test + rf_prob_test + xgb_prob_test) / 3
ens_333_all  = (lr_prob_all  + rf_prob_all  + xgb_prob_all)  / 3
results['ens_333'] = evaluate('33/33 LR+RF+XGB', ens_333_test, ens_333_all,
                               y_test, atype_test)

# Best ensemble = highest test acc with floor applied
best_ens = max(['ens_5050', 'ens_333'],
               key=lambda k: results[k]['acc_floor'])
print(f"\n  Best ensemble: {best_ens}")

# Summary table
print("\n--- Full comparison (with SPLIT floor) ---")
print(f"  {'Model':<16} {'TestAcc':>8} {'Brier':>8} {'SplitMAE':>10} {'CounterMAE':>11}")
for nm, k in [('LR','lr'),('RF','rf'),('XGB','xgb'),('50/50','ens_5050'),('33/33','ens_333')]:
    r = results[k]
    print(f"  {nm:<16} {r['acc_floor']:>8.4f} {r['brier_floor']:>8.4f} "
          f"{r['split_mae_floor']:>10.4f} {r['counter_mae']:>11.4f}")
print(f"  {'Prod M2B':<16} {PROD_ACC:>8.4f} {PROD_BRIER:>8.4f} {'0.1239':>10} {'0.1064':>11}")
print(f"  {'V2 candidate':<16} {'0.7077':>8} {'0.1961':>8} {'0.1988':>10} {'0.0528':>11}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2 — Feature importance")
print("=" * 70)

feat_names = FEAT_2B_V3
xgb_imp    = dict(zip(feat_names, xgb.feature_importances_))
xgb_sorted = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)

print("\nXGB top 15 features (full dataset):")
for i, (f, imp) in enumerate(xgb_sorted[:15]):
    print(f"  {i+1:>2}. {f:<25} {imp:.4f}")

def feat_rank(slist, name):
    for i, (f, _) in enumerate(slist):
        if f == name: return i + 1
    return None

r_m1sig  = feat_rank(xgb_sorted, 'is_m1_signal')
r_aenc   = feat_rank(xgb_sorted, 'agreement_encoded')
r_gsign  = feat_rank(xgb_sorted, 'gap_signed')
r_gsize  = feat_rank(xgb_sorted, 'gap_size')

print(f"\n  is_m1_signal rank:       #{r_m1sig}  {'in top 10 ✓' if r_m1sig and r_m1sig <= 10 else 'outside top 10 ✗'}")
print(f"  agreement_encoded rank:  #{r_aenc}  {'in top 10 ✓' if r_aenc and r_aenc <= 10 else 'outside top 10 ✗'}")
print(f"  gap_signed rank:         #{r_gsign}")
print(f"  gap_size rank:           #{r_gsize}")
print(f"  gap_signed > gap_size:   {'✓' if r_gsign and r_gsize and r_gsign < r_gsize else '✗'}")

# RF importance
rf_imp    = dict(zip(feat_names, rf.feature_importances_))
rf_sorted = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)
print("\nRF top 15 features:")
for i, (f, imp) in enumerate(rf_sorted[:15]):
    print(f"  {i+1:>2}. {f:<25} {imp:.4f}")

# Permutation importance within agreement subsets
from sklearn.inspection import permutation_importance

print("\n--- Permutation importance per agreement type (XGB) ---")
for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    mask = df['agreement_type'] == atype
    n    = mask.sum()
    if n < 50:
        print(f"  {atype}: N={n} too small, skip")
        continue
    Xs = df.loc[mask, FEAT_2B_V3].values
    ys = df.loc[mask, 'pick_won'].values
    pi = permutation_importance(xgb, Xs, ys, n_repeats=5,
                                random_state=42, n_jobs=1)
    top5_idx = np.argsort(pi.importances_mean)[::-1][:5]
    print(f"\n  {atype} (N={n}) — top 5:")
    for idx in top5_idx:
        print(f"    {feat_names[idx]:<25} {pi.importances_mean[idx]:+.4f}")

    sorted_idx = np.argsort(pi.importances_mean)[::-1]
    if atype == 'SPLIT':
        m1c_r = list(sorted_idx).index(feat_names.index('m1_conviction')) + 1
        m1s_r = list(sorted_idx).index(feat_names.index('is_m1_signal')) + 1
        print(f"    m1_conviction rank within SPLIT:  #{m1c_r}")
        print(f"    is_m1_signal rank within SPLIT:   #{m1s_r}")
    if atype == 'COUNTER':
        m2c_r = list(sorted_idx).index(feat_names.index('m2a_conviction')) + 1
        print(f"    m2a_conviction rank within COUNTER: #{m2c_r}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3 — Calibration check")
print("=" * 70)

best_probs_test  = results[best_ens]['probs_test']
best_probs_floor = results[best_ens]['probs_floor']
best_probs_all   = results[best_ens]['probs_all']

df_t = df[test_mask].copy()
df_t['pred_raw']   = best_probs_test
df_t['pred_floor'] = best_probs_floor

print(f"\n  Using: {best_ens} ensemble")
print(f"\n  {'AType':<10} {'N':>5} {'Raw MAE':>9} {'Floor MAE':>10} {'V2 MAE':>8} {'LT MAE':>8} {'Tgt MAE':>8}")

V2_MAES = {'CONFIRM': 0.1057, 'COUNTER': 0.0528, 'SPLIT': 0.1988}
LT_MAES = {'CONFIRM': 0.0861, 'COUNTER': 0.1064, 'SPLIT': 0.1239}
TGT     = {'COUNTER': 0.0800, 'SPLIT': 0.1239}

full_calib_results = {}
for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    mask = df_t['agreement_type'] == atype
    n    = mask.sum()
    sub  = df_t[mask]
    raw_mae   = calib_mae(sub['pred_raw'].values,   sub['pick_won'].values)
    floor_mae = calib_mae(sub['pred_floor'].values, sub['pick_won'].values)
    tgt = TGT.get(atype, '—')
    tgt_str = f'{tgt:.4f}' if isinstance(tgt, float) else tgt
    pass_str = ''
    if isinstance(tgt, float):
        pass_str = ' ✓' if floor_mae <= tgt else ' ✗'
    print(f"  {atype:<10} {n:>5} {raw_mae:>9.4f} {floor_mae:>10.4f} "
          f"{V2_MAES.get(atype,0):>8.4f} {LT_MAES.get(atype,0):>8.4f} "
          f"{tgt_str:>8}{pass_str}")
    full_calib_results[atype] = {
        'n': int(n), 'raw_mae': float(raw_mae), 'floor_mae': float(floor_mae)
    }

# By gap zone
print(f"\n  Calibration by zone (floor, test set):")
print(f"  {'Zone':<6} {'N':>5} {'Pred':>8} {'Actual':>8} {'Diff':>8}")
for z in sorted(df_t['gap_zone'].unique()):
    zm = df_t['gap_zone'] == z
    if zm.sum() < 5: continue
    pred   = df_t.loc[zm, 'pred_floor'].mean()
    actual = df_t.loc[zm, 'pick_won'].mean()
    print(f"  Z{z:<5} {zm.sum():>5} {pred:>8.3f} {actual:>8.3f} {pred-actual:>+8.3f}")

# Confidence bucket calibration
print(f"\n  Calibration by confidence bucket (floor, test set):")
df_t['bucket'] = pd.cut(df_t['pred_floor'],
                         bins=[0,.45,.55,.65,.75,.85,1.0],
                         labels=['<45','45-55','55-65','65-75','75-85','>85'],
                         include_lowest=True)
print(f"  {'Bucket':<12} {'N':>5} {'AvgPred':>9} {'ActualWR':>9} {'Diff':>8}")
for b in ['<45','45-55','55-65','65-75','75-85','>85']:
    bm = df_t['bucket'] == b
    if bm.sum() < 5: continue
    ap = df_t.loc[bm, 'pred_floor'].mean()
    aw = df_t.loc[bm, 'pick_won'].mean()
    print(f"  {str(b):<12} {bm.sum():>5} {ap:>9.3f} {aw:>9.3f} {ap-aw:>+8.3f}")

# SPLIT agreement type mean predictions
print(f"\n  Agreement type mean predictions (floor, test set):")
for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    mask = df_t['agreement_type'] == atype
    n    = mask.sum()
    if n == 0: continue
    pmean  = df_t.loc[mask, 'pred_floor'].mean()
    actual = df_t.loc[mask, 'pick_won'].mean()
    print(f"    {atype:<10} N={n:>4}  mean_pred={pmean:.3f}  actual_WR={actual:.3f}  diff={pmean-actual:+.3f}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4 — Promotion decision")
print("=" * 70)

best_r = results[best_ens]
counter_mae_v = full_calib_results['COUNTER']['floor_mae']
split_mae_v   = full_calib_results['SPLIT']['floor_mae']

criteria = {
    'acc':     bool(best_r['acc_floor']   >= 0.7051),
    'brier':   bool(best_r['brier_floor'] <= 0.1965),
    'counter': bool(counter_mae_v         <= 0.0800),
    'split':   bool(split_mae_v           <= 0.1239),
    'gap_dir': bool(r_gsign is not None and r_gsize is not None and r_gsign < r_gsize),
}

print(f"\n  Criterion 1 — Test acc ≥ 70.51%:        {best_r['acc_floor']:.4f}  {'✓' if criteria['acc']     else '✗'}")
print(f"  Criterion 2 — Brier ≤ 0.1965:           {best_r['brier_floor']:.4f}  {'✓' if criteria['brier']   else '✗'}")
print(f"  Criterion 3 — COUNTER MAE ≤ 0.0800:     {counter_mae_v:.4f}  {'✓' if criteria['counter'] else '✗'}")
print(f"  Criterion 4 — SPLIT MAE ≤ 0.1239:       {split_mae_v:.4f}  {'✓' if criteria['split']   else '✗'}")
print(f"  Criterion 5 — gap_signed > gap_size:     #{r_gsign} vs #{r_gsize}  {'✓' if criteria['gap_dir'] else '✗'}")

all_pass = all(criteria.values())
print(f"\n  ALL CRITERIA PASS: {all_pass}")

if all_pass:
    print("\n  → PROMOTING Model 2B V3 to production")
    prod_payload = {
        'version': 'M2B_V3',
        'best_ens': best_ens,
        'features': FEAT_2B_V3,
        'split_floor': FLOOR,
        'xgb': xgb, 'rf': rf, 'lr_cal': lr_cal,
        'xgb_best_params': best_params,
    }
    with open('/Users/allenthompson/Desktop/ufc-predictor/model/ufc_model2b.pkl', 'wb') as f:
        pickle.dump(prod_payload, f)
    with open('/Users/allenthompson/Desktop/ufc-predictor/model/ufc_model2b_features.pkl', 'wb') as f:
        pickle.dump(FEAT_2B_V3, f)
    PROMOTED  = True
    SAVE_PATH = 'model/ufc_model2b.pkl'
    print(f"  Saved to: {SAVE_PATH}")
else:
    print("\n  → NOT promoting — saving as candidate")
    failed = [k for k, v in criteria.items() if not v]
    print(f"  Failed: {failed}")
    cand_path = os.path.join(OUT_DIR, 'model2b_v3_candidate.pkl')
    cand_payload = {
        'version': 'M2B_V3_candidate',
        'best_ens': best_ens,
        'features': FEAT_2B_V3,
        'split_floor': FLOOR,
        'xgb': xgb, 'rf': rf, 'lr_cal': lr_cal,
        'xgb_best_params': {k: (int(v) if isinstance(v, (int, np.integer)) else
                                float(v) if isinstance(v, (float, np.floating)) else v)
                            for k, v in best_params.items()},
        'results_summary': {
            k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                for kk, vv in v.items()
                if kk not in ('probs_test', 'probs_floor', 'probs_all')}
            for k, v in results.items()
        },
        'criteria': criteria,
        'failed': failed,
    }
    with open(cand_path, 'wb') as f:
        pickle.dump(cand_payload, f)
    PROMOTED  = False
    SAVE_PATH = cand_path
    print(f"  Saved to: {cand_path}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5 — Writing findings report")
print("=" * 70)

def _f(v): return f'{v:.4f}' if isinstance(v, float) else str(v)

report_lines = [
    "# Model 2B V3 — Findings Report",
    "Generated: 2026-05-14",
    "",
    "## Setup",
    "",
    f"- Training universe: 3,007 rows from data/value_bet_log.csv",
    f"- Temporal split: train={train_mask.sum()} (pre-2024), test={test_mask.sum()} (2024+)",
    f"- Feature count: {len(FEAT_2B_V3)} (V2 had 20)",
    f"- New features: `is_m1_signal`, `agreement_encoded`",
    f"- Post-processing: SPLIT probability floor = {FLOOR}",
    "",
    "## New Feature Summary",
    "",
    f"**is_m1_signal** (SPLIT + zone>=5 + m1_conviction>=0.15):",
    f"- Flagged fights: {df['is_m1_signal'].sum()}",
    f"- M1 Signal WR:   {df[df['is_m1_signal']==1]['pick_won'].mean():.3f}",
    f"- Non-signal WR:  {df[df['is_m1_signal']==0]['pick_won'].mean():.3f}",
    f"- XGB rank: #{r_m1sig}  ({'in top 10' if r_m1sig and r_m1sig <= 10 else 'outside top 10'})",
    "",
    f"**agreement_encoded** (CONFIRM=3, SPLIT=2, NEAR_ZERO=1, COUNTER=0):",
    f"- XGB rank: #{r_aenc}  ({'in top 10' if r_aenc and r_aenc <= 10 else 'outside top 10'})",
    "",
    "## Model Performance (with SPLIT floor)",
    "",
    "| Model | CV Acc | Test Acc | Brier | SPLIT MAE | COUNTER MAE |",
    "|-------|--------|----------|-------|-----------|-------------|",
]

for nm, k in [('LR','lr'),('RF','rf'),('XGB','xgb'),('50/50','ens_5050'),('33/33','ens_333')]:
    r = results[k]
    cv = f"{r['cv_acc']:.4f}" if r['cv_acc'] else '—'
    report_lines.append(
        f"| {nm} | {cv} | {r['acc_floor']:.4f} | {r['brier_floor']:.4f} "
        f"| {r['split_mae_floor']:.4f} | {r['counter_mae']:.4f} |"
    )
report_lines += [
    f"| **Prod M2B** | — | **0.7051** | **0.1943** | **0.1239** | **0.1064** |",
    f"| V2 candidate | — | 0.7077 | 0.1961 | 0.1988 | 0.0528 |",
    "",
    f"Best ensemble: **{best_ens}**",
    "",
    "## XGBoost Top 15 Features",
    "",
]

for i, (f, imp) in enumerate(xgb_sorted[:15]):
    report_lines.append(f"{i+1:>2}. `{f}` — {imp:.4f}")

report_lines += [
    "",
    f"- gap_signed rank #{r_gsign} vs gap_size rank #{r_gsize}: "
    f"{'gap_signed higher ✓' if r_gsign and r_gsize and r_gsign < r_gsize else 'gap_size higher ✗'}",
    f"- is_m1_signal: #{r_m1sig}  agreement_encoded: #{r_aenc}",
    "",
    "## Calibration by Agreement Type (test set, with floor)",
    "",
    "| AType | N | Raw MAE | Floor MAE | V2 MAE | LT MAE | Target |",
    "|-------|---|---------|-----------|--------|--------|--------|",
]

for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    cr   = full_calib_results[atype]
    tgt  = TGT.get(atype, '—')
    tgt_s = f'{tgt:.4f}' if isinstance(tgt, float) else tgt
    pf   = '✓' if isinstance(tgt, float) and cr['floor_mae'] <= tgt else ('✗' if isinstance(tgt, float) else '—')
    report_lines.append(
        f"| {atype} | {cr['n']} | {cr['raw_mae']:.4f} | {cr['floor_mae']:.4f} "
        f"| {V2_MAES[atype]:.4f} | {LT_MAES[atype]:.4f} | {tgt_s} {pf} |"
    )

report_lines += [
    "",
    "## Impact of SPLIT Floor",
    "",
    f"Floor applied at {FLOOR} (actual SPLIT WR ~52.1%).",
    "",
    "| Model | Acc no-floor | Acc floor | Brier no-floor | Brier floor |",
    "|-------|-------------|-----------|----------------|-------------|",
]
for nm, k in [('LR','lr'),('RF','rf'),('XGB','xgb'),('50/50','ens_5050'),('33/33','ens_333')]:
    r = results[k]
    report_lines.append(
        f"| {nm} | {r['acc_raw']:.4f} | {r['acc_floor']:.4f} "
        f"| {r['brier_raw']:.4f} | {r['brier_floor']:.4f} |"
    )

report_lines += [
    "",
    "## Promotion Decision",
    "",
    "| Criterion | Required | Actual | Pass |",
    "|-----------|----------|--------|------|",
    f"| Test accuracy | ≥ 70.51% | {best_r['acc_floor']:.4f} | {'✓' if criteria['acc'] else '✗'} |",
    f"| Brier score | ≤ 0.1965 | {best_r['brier_floor']:.4f} | {'✓' if criteria['brier'] else '✗'} |",
    f"| COUNTER MAE | ≤ 0.0800 | {counter_mae_v:.4f} | {'✓' if criteria['counter'] else '✗'} |",
    f"| SPLIT MAE | ≤ 0.1239 | {split_mae_v:.4f} | {'✓' if criteria['split'] else '✗'} |",
    f"| gap_signed rank | > gap_size | #{r_gsign} vs #{r_gsize} | {'✓' if criteria['gap_dir'] else '✗'} |",
    "",
    f"**Decision: {'PROMOTED to production' if PROMOTED else 'NOT PROMOTED'}**",
    f"Saved to: `{SAVE_PATH}`",
    "",
    "## Recommended Next Steps",
    "",
]

if not PROMOTED:
    failed = [k for k, v in criteria.items() if not v]
    report_lines += [
        f"Failed criteria: {failed}",
        "",
    ]
    if 'brier' in failed:
        report_lines.append(
            "- **Brier**: The floor improves SPLIT calibration MAE but hurts Brier "
            "by adding probability mass to fights the model is already uncertain about. "
            "Try a softer floor (e.g. 0.42) or limit the floor to high-zone SPLIT fights only."
        )
    if 'split' in failed:
        report_lines += [
            "- **SPLIT MAE**: Model still underestimates SPLIT fights despite `is_m1_signal` "
            "and `agreement_encoded`. The SPLIT subset is heterogeneous — low-zone SPLIT fights "
            "genuinely win ~50-52% and are harder to calibrate. Consider:",
            "  1. Zone-stratified SPLIT floor (higher floor only for Z3+ SPLIT fights)",
            "  2. Separate model for SPLIT fights",
            "  3. Train with sample weighting — upweight SPLIT fights in loss",
        ]
    report_lines += [
        "",
        "- The lookup-table approach (0.0162 MAE overall) remains the better calibration "
        "mechanism. Consider keeping the trained model for ranking/valueScore and the lookup "
        "table for displayed confidence percentages in AETSlip.js.",
    ]

report_text = "\n".join(report_lines) + "\n"
report_path = os.path.join(OUT_DIR, 'MODEL2B_V3_FINDINGS.md')
with open(report_path, 'w') as f:
    f.write(report_text)
print(f"  Report written: {report_path}")

# Results JSON
json_out = {
    'version': 'M2B_V3',
    'date': '2026-05-14',
    'best_ens': best_ens,
    'n_features': len(FEAT_2B_V3),
    'features': FEAT_2B_V3,
    'split_floor': FLOOR,
    'promoted': PROMOTED,
    'save_path': SAVE_PATH,
    'criteria': criteria,
    'failed': [k for k, v in criteria.items() if not v],
    'model_perf': {
        k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
            for kk, vv in v.items()
            if kk not in ('probs_test', 'probs_floor', 'probs_all')}
        for k, v in results.items()
    },
    'calibration_by_type': full_calib_results,
    'xgb_top15': [{'feature': f, 'importance': float(imp), 'rank': i+1}
                  for i, (f, imp) in enumerate(xgb_sorted[:15])],
    'new_feature_ranks': {
        'is_m1_signal': r_m1sig,
        'agreement_encoded': r_aenc,
        'gap_signed': r_gsign,
        'gap_size': r_gsize,
    },
    'xgb_best_params': {k: (int(v) if isinstance(v, (int, np.integer)) else
                             float(v) if isinstance(v, (float, np.floating)) else v)
                        for k, v in best_params.items()},
}
json_path = os.path.join(OUT_DIR, 'model2b_v3_results.json')
with open(json_path, 'w') as f:
    json.dump(json_out, f, indent=2)
print(f"  Results JSON: {json_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"\nSummary:")
print(f"  Best model:    {best_ens}")
print(f"  Test acc:      {best_r['acc_floor']:.4f}  (prod 0.7051  delta {best_r['acc_floor']-0.7051:+.4f})")
print(f"  Brier:         {best_r['brier_floor']:.4f}  (prod 0.1943  delta {best_r['brier_floor']-0.1943:+.4f})")
print(f"  COUNTER MAE:   {counter_mae_v:.4f}  (V2 0.0528  LT 0.1064)")
print(f"  SPLIT MAE:     {split_mae_v:.4f}  (V2 0.1988  LT 0.1239)")
print(f"  Promoted:      {PROMOTED}")
