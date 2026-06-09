#!/usr/bin/env python3
"""
fix_and_retrain_womens_m2a.py
Step 1 — Confirm corner bias in value_bet_log_womens.csv
Step 2 — Randomize corners (seed=42), save data/value_bet_log_womens_rand.csv
Step 3 — Retrain Women's M2A on randomized log

Key insight: UFC books Vegas favorites in the Red corner ~66% of the time.
Without randomization the model learns red=favorite, inflating apparent accuracy.
After randomization the target is ~50/50 and the model must use actual features.

Save rule:
  If randomized model temporal accuracy > Women's M1 (74.30%):
      overwrite model/ufc_model2a_womens_lr.pkl + xgb.pkl
  Else:
      save as model/ufc_model2a_womens_lr_rand_v2.pkl + xgb.pkl
      print: "Women's M2A does not beat M1 — routing through M1 only"
"""
import gc, json, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.chdir('/Users/allenthompson/Desktop/ufc-predictor')

SEED           = 42
TRAIN_CUTOFF   = pd.Timestamp('2024-01-01')
N_TRIALS       = 50
W_M1_ACC       = 0.7430   # women's M1 threshold
BIASED_ACC     = 0.6838   # previous biased run
MEN_M2A_ACC    = 0.7320
WOMENS_CLASSES = ["Women's Strawweight","Women's Flyweight",
                  "Women's Bantamweight","Women's Featherweight"]
METHOD_COLS    = ['r_ko_odds','b_ko_odds','r_sub_odds','b_sub_odds','r_dec_odds','b_dec_odds']

TIER_FALLBACK = {'hfav':0.91,'mfav':0.82,'sfav':0.73,
                 'pkem':0.59,'sdog':0.47,'mdog':0.33,'hdog':0.18}

print("=" * 70)
print("STEP 1 — CONFIRM CORNER BIAS")
print("=" * 70)

log = pd.read_csv('data/value_bet_log_womens.csv')
log['date'] = pd.to_datetime(log['date'])

# Red corner wins (from log perspective: m1_prob > 50 → M1 picked F1=Red)
red_wins = np.where(log['m1_prob'] > 50, log['pick_won'], 1 - log['pick_won'])
print(f"\n  Red corner win rate (log) : {red_wins.mean()*100:.2f}%  (N={len(log):,})")
print(f"  Expected after fix        : ~50.00%")
print(f"\n  By agreement type:")
for at in ['CONFIRM_DOG','CONFIRM_FAV','NO_EDGE']:
    mask = log['agreement_type'] == at
    rw = np.where(log.loc[mask,'m1_prob'] > 50,
                  log.loc[mask,'pick_won'], 1 - log.loc[mask,'pick_won'])
    print(f"    {at:<15}: {rw.mean()*100:.2f}%  (N={mask.sum()})")

master_full = pd.read_csv('data/ufc-master.csv', low_memory=False)
master_full['date'] = pd.to_datetime(master_full['date'])
wf = master_full[master_full['weight_class'].isin(WOMENS_CLASSES) &
                 master_full['R_odds'].notna() & master_full['B_odds'].notna() &
                 (master_full['R_odds'] != 0) & (master_full['B_odds'] != 0)].copy()
r_is_fav = (pd.to_numeric(wf['R_odds'], errors='coerce') < 0)
print(f"\n  ufc-master.csv women's fights with ML odds: {len(wf):,}")
print(f"    Red corner = Vegas favorite : {r_is_fav.mean()*100:.1f}%")
print(f"    Red corner wins overall     : {(wf['Winner']=='Red').mean()*100:.1f}%")
print(f"\n  Root cause: UFC books favorites in the Red corner ~{r_is_fav.mean()*100:.0f}% of the time.")
print(f"  Without randomization, the model learns red≈favorite, inflating accuracy.")
del wf
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — CORNER RANDOMIZATION (seed=42)")
print("=" * 70)

rand_log = log.copy()
np.random.seed(SEED)
swap_mask = np.random.random(len(rand_log)) < 0.5
n_swap = swap_mask.sum()
print(f"\n  Rows to swap: {n_swap:,} / {len(rand_log):,}  ({n_swap/len(rand_log)*100:.1f}%)")

# For swapped rows: f1↔f2, m1_prob → 100-m1_prob, m2a_prob → 100-m2a_prob
# All pick-relative columns stay the same (gap, pick_novig, pick_won, vegas_agree, etc.)
idx = rand_log.index[swap_mask]
rand_log.loc[idx, ['f1_name','f2_name']] = rand_log.loc[idx, ['f2_name','f1_name']].values
rand_log.loc[idx, 'm1_prob']  = (100.0 - rand_log.loc[idx, 'm1_prob']).values
rand_log.loc[idx, 'm2a_prob'] = (100.0 - rand_log.loc[idx, 'm2a_prob']).values

# Store swap flag for method-odds alignment during training
rand_log['corner_swapped'] = swap_mask.astype(int)

# Verify
red_wins_after = np.where(rand_log['m1_prob'] > 50,
                           rand_log['pick_won'], 1 - rand_log['pick_won'])
print(f"\n  Red corner win rate BEFORE randomization: {red_wins.mean()*100:.2f}%")
print(f"  Red corner win rate AFTER  randomization: {red_wins_after.mean()*100:.2f}%  ✓")
print(f"\n  m1_prob distribution after randomization:")
print(f"    mean={rand_log['m1_prob'].mean():.2f}  median={rand_log['m1_prob'].median():.2f}  "
      f"pct>50: {(rand_log['m1_prob']>50).mean()*100:.1f}%")

rand_log.to_csv('data/value_bet_log_womens_rand.csv', index=False)
print(f"\n  Saved → data/value_bet_log_womens_rand.csv  ({len(rand_log):,} rows)")
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — RETRAIN WOMEN'S M2A ON RANDOMIZED LOG")
print("=" * 70)

# ── Load randomized log + merge method odds ───────────────────────────────────
print("\nLoading randomized log + merging method odds...")

master_slim = master_full[['R_fighter','B_fighter','date','R_odds','B_odds'] +
                           METHOD_COLS].copy()

# Merge on original corners (master always has R_fighter = original red)
# f1_name may now be the original B_fighter (when corner_swapped=1)
# We need both original corner sets → merge on BOTH orderings
master_r = master_slim.rename(columns={'R_fighter':'f1_name','B_fighter':'f2_name',
                                        'R_odds':'f1_odds_orig','B_odds':'f2_odds_orig',
                                        **{c: f'orig_{c}' for c in METHOD_COLS}})
master_b = master_slim.rename(columns={'B_fighter':'f1_name','R_fighter':'f2_name',
                                        'B_odds':'f1_odds_orig','R_odds':'f2_odds_orig',
                                        'r_ko_odds':'orig_b_ko_odds','b_ko_odds':'orig_r_ko_odds',
                                        'r_sub_odds':'orig_b_sub_odds','b_sub_odds':'orig_r_sub_odds',
                                        'r_dec_odds':'orig_b_dec_odds','b_dec_odds':'orig_r_dec_odds'})

# Use corner_swapped to pick the right merge source
not_swapped = rand_log[rand_log['corner_swapped'] == 0].copy()
swapped     = rand_log[rand_log['corner_swapped'] == 1].copy()

ns_merged = not_swapped.merge(master_r, on=['f1_name','f2_name','date'], how='left')
sw_merged = swapped.merge(master_b, on=['f1_name','f2_name','date'], how='left')
merged = pd.concat([ns_merged, sw_merged], ignore_index=True).sort_values('date').reset_index(drop=True)

# Rename orig_* to final method col names
method_rename = {f'orig_{c}':c for c in METHOD_COLS}
# For swapped set, master_b already put them in the right order (orig_b_ko_odds → orig_r_ko_odds etc.)
# Let's just unify to consistent names
merged = merged.rename(columns={
    'orig_r_ko_odds':'r_ko_odds','orig_b_ko_odds':'b_ko_odds',
    'orig_r_sub_odds':'r_sub_odds','orig_b_sub_odds':'b_sub_odds',
    'orig_r_dec_odds':'r_dec_odds','orig_b_dec_odds':'b_dec_odds',
})

n_before = len(merged)
merged = merged[merged[METHOD_COLS].notna().all(axis=1)].copy().reset_index(drop=True)
print(f"  Before method filter: {n_before:,}  |  After: {len(merged):,}")
print(f"  Train (<2024): {(merged['date'] < TRAIN_CUTOFF).sum()}  "
      f"Test (2024+): {(merged['date'] >= TRAIN_CUTOFF).sum()}")
gc.collect()

# ── Feature engineering ───────────────────────────────────────────────────────
print("\nEngineering features...")

def implied(odds):
    o = pd.to_numeric(odds, errors='coerce').fillna(0.0)
    return np.where(o==0, 0.0, np.where(o<0, (-o)/(-o+100), 100/(o+100)))

# 6-way no-vig — after corner alignment, r_ko_odds = odds for F1 to win by KO
r_ko_r  = implied(merged['r_ko_odds']); b_ko_r  = implied(merged['b_ko_odds'])
r_sub_r = implied(merged['r_sub_odds']); b_sub_r = implied(merged['b_sub_odds'])
r_dec_r = implied(merged['r_dec_odds']); b_dec_r = implied(merged['b_dec_odds'])

total6 = r_ko_r + b_ko_r + r_sub_r + b_sub_r + r_dec_r + b_dec_r
total6 = np.where(total6 <= 0, 1.0, total6)
r_ko  = r_ko_r/total6; b_ko  = b_ko_r/total6
r_sub = r_sub_r/total6; b_sub = b_sub_r/total6
r_dec = r_dec_r/total6; b_dec = b_dec_r/total6

merged['r_ko_prob']  = r_ko; merged['b_ko_prob']  = b_ko
merged['r_sub_prob'] = r_sub; merged['b_sub_prob'] = b_sub
merged['r_dec_prob'] = r_dec; merged['b_dec_prob'] = b_dec
merged['ko_diff']    = r_ko  - b_ko
merged['sub_diff']   = r_sub - b_sub
merged['dec_diff']   = r_dec - b_dec
merged['m1_prob_d']  = merged['m1_prob'] / 100.0
merged['gap_d']      = merged['gap'] / 100.0
merged['gap_direction'] = merged['gap_direction'].astype(float)
merged['gap_signed'] = merged['gap_d'] * merged['gap_direction']
merged['vegas_agree']= merged['vegas_agree'].astype(float)
merged['gap_zone_f'] = merged['gap_zone'].astype(float)

# ── Tier historical win rate (randomized log, no leakage) ─────────────────────
print("Computing tier_hist_win_rate (randomized log, expanding window)...")

def pick_tier(p_pct):
    p = p_pct / 100.0
    if p > 0.75: return 'hfav'
    if p > 0.60: return 'mfav'
    if p > 0.525: return 'sfav'
    if p >= 0.475: return 'pkem'
    if p >= 0.40: return 'sdog'
    if p >= 0.25: return 'mdog'
    return 'hdog'

tier_labels = merged['pick_novig'].apply(pick_tier).tolist()
sort_order  = np.argsort(merged['date'].values, kind='stable')
tier_counts = {}; tier_wins = {}
tier_hist_wr = np.zeros(len(merged))
for k in sort_order:
    t = tier_labels[k]; c = tier_counts.get(t,0); w = tier_wins.get(t,0)
    tier_hist_wr[k] = (w/c) if c >= 5 else TIER_FALLBACK.get(t, 0.60)
    tier_counts[t] = c + 1
    tier_wins[t]   = w + merged['pick_won'].iloc[k]
merged['tier_hist_win_rate'] = tier_hist_wr

print(f"  Tier win rates (from randomized log):")
for t, cnt in tier_counts.items():
    wr = tier_wins[t]/cnt if cnt > 0 else 0
    print(f"    {t:<6}: N={cnt:3d}  WR={wr:.3f}")

# ── Target: F1 (current red corner) wins ──────────────────────────────────────
# After randomization, m1_prob > 50 → M1 picks the current F1 (now balanced ~50%)
target = np.where(merged['m1_prob'] > 50,
                  merged['pick_won'].values,
                  1 - merged['pick_won'].values).astype(int)
print(f"\n  Target (current F1 wins): {target.mean():.3f}  "
      f"(should be ~0.50 after randomization)")

FEAT_COLS = ['m1_prob_d',
             'r_ko_prob','b_ko_prob','r_sub_prob','b_sub_prob','r_dec_prob','b_dec_prob',
             'ko_diff','sub_diff','dec_diff',
             'gap_d','gap_signed','gap_zone_f',
             'vegas_agree','tier_hist_win_rate']
FEAT_NAMES = ['m1_prob',
              'r_ko_prob','b_ko_prob','r_sub_prob','b_sub_prob','r_dec_prob','b_dec_prob',
              'ko_diff','sub_diff','dec_diff',
              'gap','gap_signed','gap_zone',
              'vegas_agree','tier_hist_win_rate']

X = np.nan_to_num(merged[FEAT_COLS].values.astype(float), nan=0.0)
train_mask = (merged['date'] < TRAIN_CUTOFF).values
test_mask  = ~train_mask
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = target[train_mask], target[test_mask]
print(f"\n  Feature matrix: {X.shape}  Train: {X_train.shape}  Test: {X_test.shape}")
print(f"  Train balance: {y_train.mean():.3f}  Test balance: {y_test.mean():.3f}")
gc.collect()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Step 3A — Logistic Regression ─────────────────────────────────────────────
print(f"\nStep 3A — Logistic Regression ({N_TRIALS} Optuna trials)...")

def lr_objective(trial):
    C   = trial.suggest_float('C', 0.001, 50.0, log=True)
    pen = trial.suggest_categorical('penalty', ['l1','l2','elasticnet'])
    sc  = trial.suggest_categorical('scaler', ['robust','standard'])
    cw  = trial.suggest_categorical('class_weight', ['none','balanced'])
    cw_v = None if cw == 'none' else 'balanced'
    if pen == 'elasticnet':
        l1r = trial.suggest_float('l1_ratio', 0.0, 1.0)
        clf = LogisticRegression(C=C, penalty='elasticnet', l1_ratio=l1r,
                                 solver='saga', class_weight=cw_v, max_iter=3000, random_state=SEED)
    elif pen == 'l1':
        clf = LogisticRegression(C=C, penalty='l1', solver='saga',
                                 class_weight=cw_v, max_iter=3000, random_state=SEED)
    else:
        clf = LogisticRegression(C=C, penalty='l2', solver='saga',
                                 class_weight=cw_v, max_iter=3000, random_state=SEED)
    scaler = RobustScaler() if sc == 'robust' else StandardScaler()
    pipe   = Pipeline([('sc', scaler), ('clf', clf)])
    oof    = cross_val_predict(pipe, X_train, y_train, cv=skf,
                               method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof > 0.5).astype(int))

study_lr = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_lr.optimize(lr_objective, n_trials=N_TRIALS, show_progress_bar=False)
p_lr = study_lr.best_params

cw_v = None if p_lr['class_weight'] == 'none' else 'balanced'
pen  = p_lr['penalty']
if pen == 'elasticnet':
    clf_lr = LogisticRegression(C=p_lr['C'], penalty='elasticnet',
                                l1_ratio=p_lr.get('l1_ratio',0.5), solver='saga',
                                class_weight=cw_v, max_iter=3000, random_state=SEED)
elif pen == 'l1':
    clf_lr = LogisticRegression(C=p_lr['C'], penalty='l1', solver='saga',
                                class_weight=cw_v, max_iter=3000, random_state=SEED)
else:
    clf_lr = LogisticRegression(C=p_lr['C'], penalty='l2', solver='saga',
                                class_weight=cw_v, max_iter=3000, random_state=SEED)
sc_lr    = RobustScaler() if p_lr['scaler'] == 'robust' else StandardScaler()
model_lr = Pipeline([('sc', sc_lr), ('clf', clf_lr)])
model_lr.fit(X_train, y_train)
p_lr_test = model_lr.predict_proba(X_test)[:, 1]
acc_lr    = accuracy_score(y_test, (p_lr_test > 0.5).astype(int))
brier_lr  = brier_score_loss(y_test, p_lr_test)
auc_lr    = roc_auc_score(y_test, p_lr_test)
print(f"  LR : acc={acc_lr:.4f}  brier={brier_lr:.4f}  AUC={auc_lr:.4f}")
print(f"  Params: {p_lr}")
gc.collect()

# ── Step 3B — XGBoost ─────────────────────────────────────────────────────────
print(f"\nStep 3B — XGBoost ({N_TRIALS} Optuna trials)...")

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 50, 600),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 2, 5),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma':            trial.suggest_float('gamma', 0, 3),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0, 2),
    }
    clf = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss',
                        random_state=SEED, verbosity=0, n_jobs=1)
    oof = cross_val_predict(clf, X_train, y_train, cv=skf,
                            method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof > 0.5).astype(int))

study_xgb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
p_xgb = study_xgb.best_params

model_xgb = XGBClassifier(**p_xgb, use_label_encoder=False, eval_metric='logloss',
                           random_state=SEED, verbosity=0, n_jobs=1)
model_xgb.fit(X_train, y_train)
p_xgb_test = model_xgb.predict_proba(X_test)[:, 1]
acc_xgb    = accuracy_score(y_test, (p_xgb_test > 0.5).astype(int))
brier_xgb  = brier_score_loss(y_test, p_xgb_test)
auc_xgb    = roc_auc_score(y_test, p_xgb_test)
print(f"  XGB: acc={acc_xgb:.4f}  brier={brier_xgb:.4f}  AUC={auc_xgb:.4f}")
print(f"  Params: {p_xgb}")
gc.collect()

# ── 50/50 Blend ───────────────────────────────────────────────────────────────
print("\nStep 3C — 50/50 Blend...")
p_blend     = 0.50 * p_lr_test + 0.50 * p_xgb_test
acc_blend   = accuracy_score(y_test, (p_blend > 0.5).astype(int))
brier_blend = brier_score_loss(y_test, p_blend)
auc_blend   = roc_auc_score(y_test, p_blend)

bins = np.linspace(0, 1, 11)
cal_mae = 0.0; n_bins = 0
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_blend >= lo) & (p_blend < hi)
    if mask.sum() >= 3:
        cal_mae += abs(p_blend[mask].mean() - y_test[mask].mean())
        n_bins  += 1
cal_mae = cal_mae / n_bins if n_bins > 0 else float('nan')

print(f"  Blend: acc={acc_blend:.4f}  brier={brier_blend:.4f}  "
      f"AUC={auc_blend:.4f}  CalMAE={cal_mae:.4f}")
beats_m1 = acc_blend > W_M1_ACC
print(f"  vs Women's M1 ({W_M1_ACC:.4f}): {'BEATS ✓' if beats_m1 else 'does NOT beat ✗'}")
print(f"  vs biased run ({BIASED_ACC:.4f}): "
      f"{'+' if acc_blend >= BIASED_ACC else ''}{(acc_blend-BIASED_ACC)*100:.2f}pp")

# ── Feature importances ───────────────────────────────────────────────────────
print("\nFeature importances — XGB (top 15):")
fi_xgb = sorted(zip(FEAT_NAMES, model_xgb.feature_importances_), key=lambda x: -x[1])
for rank, (feat, imp) in enumerate(fi_xgb[:15], 1):
    print(f"  {rank:2d}. {feat:<22}: {imp:.4f}")

print("\nFeature importances — LR |coef| (top 15):")
try:
    coefs = abs(model_lr.named_steps['clf'].coef_[0])
    fi_lr = sorted(zip(FEAT_NAMES, coefs), key=lambda x: -x[1])
    for rank, (feat, imp) in enumerate(fi_lr[:15], 1):
        print(f"  {rank:2d}. {feat:<22}: {imp:.4f}")
except Exception as e:
    print(f"  (failed: {e})")

# ── ROI simulation ────────────────────────────────────────────────────────────
print("\nROI simulation (test set 2024+)...")
test_rows = merged[test_mask].copy().reset_index(drop=True)
test_rows['m2a_blend_prob'] = p_blend
test_rows['y_true']         = y_test

def american_to_decimal(ml):
    ml = float(ml)
    if ml == 0 or np.isnan(ml): return None
    return 1 + 100/abs(ml) if ml < 0 else 1 + ml/100

def sim_flat(df_sub, prob_col, stake=20):
    payouts = []; wins = 0; bets = 0
    for _, r in df_sub.iterrows():
        m2a_p    = float(r[prob_col])
        pick_f1  = m2a_p > 0.5
        # f1_odds_orig = odds for current F1 (already corner-adjusted)
        ml = r.get('f1_odds_orig', np.nan) if pick_f1 else r.get('f2_odds_orig', np.nan)
        if pd.isna(ml) or ml == 0: continue
        dec = american_to_decimal(ml)
        if dec is None: continue
        won = (int(r['y_true']) == 1) if pick_f1 else (int(r['y_true']) == 0)
        bets += 1
        payouts.append(stake * (dec-1) if won else -stake)
        if won: wins += 1
    if not payouts: return 0, 0.0, 0.0
    return bets, wins/bets*100, sum(payouts)/(stake*len(payouts))*100

def sim_qkelly(df_sub, prob_col, bankroll=1000):
    bk = bankroll; wins = 0; bets = 0
    for _, r in df_sub.iterrows():
        m2a_p   = float(r[prob_col])
        pick_f1 = m2a_p > 0.5
        p = m2a_p if pick_f1 else 1.0 - m2a_p
        ml = r.get('f1_odds_orig', np.nan) if pick_f1 else r.get('f2_odds_orig', np.nan)
        if pd.isna(ml) or ml == 0: continue
        dec = american_to_decimal(ml)
        if dec is None: continue
        b = dec - 1; q = 1 - p
        kelly = max(0.0, (b*p - q)/b) * 0.25
        won = (int(r['y_true']) == 1) if pick_f1 else (int(r['y_true']) == 0)
        bk += kelly*bk*b if won else -(kelly*bk)
        bets += 1
        if won: wins += 1
    roi = (bk - bankroll)/bankroll*100
    return bets, wins/bets*100 if bets else 0.0, roi, bk

print(f"\n  {'Group':<15}  {'N':>4}  {'Flat WR':>8}  {'Flat ROI':>9}  {'QK ROI':>9}  {'QK End $':>9}")
print(f"  {'-'*15}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")
for at in ['ALL','CONFIRM_DOG','CONFIRM_FAV','NO_EDGE']:
    sub = test_rows if at == 'ALL' else test_rows[test_rows['agreement_type'] == at]
    if len(sub) == 0: continue
    n, wr, roi = sim_flat(sub, 'm2a_blend_prob')
    qn, qwr, qroi, qend = sim_qkelly(sub, 'm2a_blend_prob')
    print(f"  {at:<15}  {n:>4}  {wr:>7.1f}%  {roi:>+8.1f}%  {qroi:>+8.1f}%  ${qend:>8,.0f}")
gc.collect()

# ── Summary comparison ────────────────────────────────────────────────────────
w_m1_test_acc = accuracy_score(y_test,
    (merged.loc[test_mask, 'm1_prob'].values > 50).astype(int))

print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)
print(f"\n  {'Model':<38}  {'Acc':>7}  {'Brier':>7}  {'AUC':>7}  {'CalMAE':>8}")
print(f"  {'-'*38}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
print(f"  {'Biased run (no randomization)':38}  {BIASED_ACC:7.4f}  {'—':>7}  {'—':>7}  {'—':>8}")
print(f"  {'Women M1 (filtered, this test set)':38}  {w_m1_test_acc:7.4f}  {'—':>7}  {'—':>7}  {'—':>8}")
print(f"  {'Women M1 full test acc':38}  {W_M1_ACC:7.4f}  {'—':>7}  {'—':>7}  {'—':>8}")
print(f"  {'Men M2A prod':38}  {MEN_M2A_ACC:7.4f}  {'—':>7}  {'—':>7}  {'—':>8}")
print(f"  {'LR (randomized)':38}  {acc_lr:7.4f}  {brier_lr:7.4f}  {auc_lr:7.4f}  {'—':>8}")
print(f"  {'XGB (randomized)':38}  {acc_xgb:7.4f}  {brier_xgb:7.4f}  {auc_xgb:7.4f}  {'—':>8}")
print(f"  {'50/50 Blend (randomized)':38}  {acc_blend:7.4f}  {brier_blend:7.4f}  {auc_blend:7.4f}  {cal_mae:8.4f}")

d_bias = acc_blend - BIASED_ACC
d_m1f  = acc_blend - w_m1_test_acc
d_m1   = acc_blend - W_M1_ACC
print(f"\n  vs biased run       : {'+' if d_bias >= 0 else ''}{d_bias*100:.2f}pp")
print(f"  vs Women M1 (filter): {'+' if d_m1f  >= 0 else ''}{d_m1f*100:.2f}pp")
print(f"  vs Women M1 (full)  : {'+' if d_m1   >= 0 else ''}{d_m1*100:.2f}pp")

# ── Save decision ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVE DECISION")
print("=" * 70)

meta = {
    'created':           datetime.now().isoformat(),
    'description':       'Women M2A — method odds + M1, corner-randomized (seed=42)',
    'features':          FEAT_NAMES, 'n_features': len(FEAT_NAMES),
    'blend':             '50% LR + 50% XGB',
    'train_size':        int(X_train.shape[0]),
    'test_size':         int(X_test.shape[0]),
    'temporal_split':    '2024-01-01',
    'corner_randomized': True, 'rand_seed': 42,
    'acc_lr': float(acc_lr), 'acc_xgb': float(acc_xgb), 'acc_blend': float(acc_blend),
    'brier_blend': float(brier_blend), 'auc_blend': float(auc_blend),
    'cal_mae_blend': float(cal_mae),
    'w_m1_acc_filtered': float(w_m1_test_acc), 'w_m1_acc_full': W_M1_ACC,
    'biased_run_acc': BIASED_ACC, 'beats_w_m1': bool(beats_m1),
    'lr_params':  {k: (float(v) if isinstance(v,float) else v) for k,v in p_lr.items()},
    'xgb_params': {k: (float(v) if isinstance(v,float) else v) for k,v in p_xgb.items()},
    'top5_features_xgb': [feat for feat, _ in fi_xgb[:5]],
}

if beats_m1:
    print(f"\n  ✓ Randomized model ({acc_blend:.4f}) BEATS Women's M1 ({W_M1_ACC:.4f})")
    print(f"  → Overwriting production model files")
    joblib.dump(model_lr,   'model/ufc_model2a_womens_lr.pkl')
    joblib.dump(model_xgb,  'model/ufc_model2a_womens_xgb.pkl')
    joblib.dump(FEAT_NAMES, 'model/feature_columns_2a_womens.pkl')
    meta['save_status'] = 'production'
    print("  Saved: model/ufc_model2a_womens_lr.pkl")
    print("  Saved: model/ufc_model2a_womens_xgb.pkl")
    print("  Saved: model/feature_columns_2a_womens.pkl")
else:
    print(f"\n  ✗ Randomized model ({acc_blend:.4f}) does NOT beat Women's M1 ({W_M1_ACC:.4f})")
    print(f"  → Saving with _rand_v2 suffix (NOT overwriting production files)")
    joblib.dump(model_lr,   'model/ufc_model2a_womens_lr_rand_v2.pkl')
    joblib.dump(model_xgb,  'model/ufc_model2a_womens_xgb_rand_v2.pkl')
    joblib.dump(FEAT_NAMES, 'model/feature_columns_2a_womens_rand_v2.pkl')
    meta['save_status'] = 'experimental_rand_v2'
    print("  Saved: model/ufc_model2a_womens_lr_rand_v2.pkl")
    print("  Saved: model/ufc_model2a_womens_xgb_rand_v2.pkl")
    print("  Saved: model/feature_columns_2a_womens_rand_v2.pkl")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Women's M2A does not beat M1 — routing women's fights      │")
    print("  │  through M1 only until more data available.                 │")
    print("  └─────────────────────────────────────────────────────────────┘")

with open('model/model2a_womens_rand_v2_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\n  Saved: model/model2a_womens_rand_v2_metadata.json")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
