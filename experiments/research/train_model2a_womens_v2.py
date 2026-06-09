#!/usr/bin/env python3
"""
train_model2a_womens_v2.py
Women's M2A — method odds + M1 signal (no moneyline features).

Data  : data/value_bet_log_womens.csv merged with ufc-master.csv method odds
Target: Red corner wins (reconstructed from pick_won + m2a_pick)
Feats : m1_prob, 6 no-vig method probs, 3 diffs, gap, gap_signed,
        gap_zone, vegas_agree, tier_hist_win_rate (women's only)
Blend : 50% LR + 50% XGB  |  Optuna 50 trials each  |  n_jobs=1

Save  : model/ufc_model2a_womens_lr.pkl
        model/ufc_model2a_womens_xgb.pkl
        model/feature_columns_2a_womens.pkl
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

SEED         = 42
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
N_TRIALS     = 50
MEN_M2A_ACC  = 0.7320   # men's M2A baseline for side-by-side
W_M1_ACC     = 0.7430   # women's M1 temporal accuracy

METHOD_COLS = ['r_ko_odds','b_ko_odds','r_sub_odds','b_sub_odds','r_dec_odds','b_dec_odds']

TIER_FALLBACK = {
    'hfav':0.91, 'mfav':0.82, 'sfav':0.73,
    'pkem':0.59, 'sdog':0.47, 'mdog':0.33, 'hdog':0.18,
}

print("=" * 70)
print("WOMEN'S M2A — METHOD ODDS + M1 SIGNAL")
print("=" * 70)

# ─── Load women's log ─────────────────────────────────────────────────────────
print("\nLoading women's value bet log...")
log = pd.read_csv('data/value_bet_log_womens.csv')
log['date'] = pd.to_datetime(log['date'])
print(f"  {len(log):,} rows  |  {log['date'].min().date()} → {log['date'].max().date()}")
print(f"  Train (<2024): {(log['split']=='train').sum()}  Test (2024+): {(log['split']=='test').sum()}")

# ─── Merge method odds from ufc-master.csv ────────────────────────────────────
print("\nMerging method odds from ufc-master.csv...")
master = pd.read_csv('data/ufc-master.csv', low_memory=False)[
    ['R_fighter','B_fighter','date'] + METHOD_COLS + ['R_odds','B_odds']
].copy()
master['date'] = pd.to_datetime(master['date'])

merged = log.merge(
    master.rename(columns={'R_fighter':'f1_name','B_fighter':'f2_name'}),
    on=['f1_name','f2_name','date'], how='left'
)
n_before = len(merged)
merged = merged[merged[METHOD_COLS].notna().all(axis=1)].copy().reset_index(drop=True)
print(f"  Before method filter: {n_before}  |  After: {len(merged):,}")
print(f"  Train: {(merged['date'] < TRAIN_CUTOFF).sum()}  Test: {(merged['date'] >= TRAIN_CUTOFF).sum()}")
gc.collect()

# ─── Feature engineering ──────────────────────────────────────────────────────
print("\nEngineering features...")

def implied(odds):
    o = pd.to_numeric(odds, errors='coerce').fillna(0.0)
    return np.where(o==0, 0.0,
           np.where(o < 0, (-o)/(-o+100), 100/(o+100)))

# 6-way no-vig method probs
r_ko_r  = implied(merged['r_ko_odds']);  b_ko_r  = implied(merged['b_ko_odds'])
r_sub_r = implied(merged['r_sub_odds']); b_sub_r = implied(merged['b_sub_odds'])
r_dec_r = implied(merged['r_dec_odds']); b_dec_r = implied(merged['b_dec_odds'])

total6 = r_ko_r + b_ko_r + r_sub_r + b_sub_r + r_dec_r + b_dec_r
total6 = np.where(total6 <= 0, 1.0, total6)

r_ko  = r_ko_r  / total6; b_ko  = b_ko_r  / total6
r_sub = r_sub_r / total6; b_sub = b_sub_r / total6
r_dec = r_dec_r / total6; b_dec = b_dec_r / total6

merged['r_ko_prob']  = r_ko;  merged['b_ko_prob']  = b_ko
merged['r_sub_prob'] = r_sub; merged['b_sub_prob'] = b_sub
merged['r_dec_prob'] = r_dec; merged['b_dec_prob'] = b_dec
merged['ko_diff']    = r_ko  - b_ko
merged['sub_diff']   = r_sub - b_sub
merged['dec_diff']   = r_dec - b_dec

# Convert log percentages to decimals
merged['m1_prob_d']    = merged['m1_prob'] / 100.0
merged['gap_d']        = merged['gap'] / 100.0
merged['gap_direction']= merged['gap_direction'].astype(float)
merged['gap_signed']   = merged['gap_d'] * merged['gap_direction']   # |gap| always ≥ 0
merged['vegas_agree']  = merged['vegas_agree'].astype(float)
merged['gap_zone_f']   = merged['gap_zone'].astype(float)

# ─── Tier historical win rate (women's only, no leakage) ──────────────────────
print("Computing tier_hist_win_rate (women's log only, expanding window)...")

def pick_tier(pick_novig_pct):
    p = pick_novig_pct / 100.0
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
    t = tier_labels[k]
    c = tier_counts.get(t, 0); w = tier_wins.get(t, 0)
    tier_hist_wr[k] = (w/c) if c >= 5 else TIER_FALLBACK.get(t, 0.60)
    tier_counts[t] = c + 1
    tier_wins[t]   = w + merged['pick_won'].iloc[k]

merged['tier_hist_win_rate'] = tier_hist_wr
print(f"  Tier win rates built (cold-start fallback for N<5)")

# Final tier breakdown
for t, cnt in tier_counts.items():
    wr = tier_wins[t]/cnt if cnt > 0 else 0
    print(f"  {t:<6}: N={cnt:3d}  WR={wr:.3f}")

# ─── Target: Red corner wins ──────────────────────────────────────────────────
# m1_prob > 50 → M1 picks F1 (Red) → pick_won = Red won
# m1_prob ≤ 50 → M1 picks F2 (Blue) → pick_won = 1 means Blue won = Red LOST
target = np.where(merged['m1_prob'] > 50,
                  merged['pick_won'].values,
                  1 - merged['pick_won'].values).astype(int)
print(f"\n  Target (Red wins): {target.mean():.3f}  (should be ~0.5 after random corner assmt)")

# ─── Feature matrix ───────────────────────────────────────────────────────────
FEAT_COLS = [
    'm1_prob_d',
    'r_ko_prob', 'b_ko_prob', 'r_sub_prob', 'b_sub_prob', 'r_dec_prob', 'b_dec_prob',
    'ko_diff', 'sub_diff', 'dec_diff',
    'gap_d', 'gap_signed', 'gap_zone_f',
    'vegas_agree',
    'tier_hist_win_rate',
]
# Canonical names for saved feature list (cleaner, no _d suffix)
FEAT_NAMES = [
    'm1_prob',
    'r_ko_prob','b_ko_prob','r_sub_prob','b_sub_prob','r_dec_prob','b_dec_prob',
    'ko_diff','sub_diff','dec_diff',
    'gap','gap_signed','gap_zone',
    'vegas_agree',
    'tier_hist_win_rate',
]

X = merged[FEAT_COLS].values.astype(float)
X = np.nan_to_num(X, nan=0.0)
print(f"\n  Feature matrix: {X.shape}  features: {FEAT_NAMES}")

train_mask = (merged['date'] < TRAIN_CUTOFF).values
test_mask  = ~train_mask
X_train, X_test   = X[train_mask], X[test_mask]
y_train, y_test   = target[train_mask], target[test_mask]
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
print(f"  Train class balance: {y_train.mean():.3f}  Test: {y_test.mean():.3f}")
gc.collect()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ─── STEP A — Logistic Regression (Optuna 50 trials) ─────────────────────────
print(f"\nStep A — Logistic Regression ({N_TRIALS} Optuna trials)...")

def lr_objective(trial):
    C    = trial.suggest_float('C', 0.001, 50.0, log=True)
    pen  = trial.suggest_categorical('penalty', ['l1','l2','elasticnet'])
    sc   = trial.suggest_categorical('scaler', ['robust','standard'])
    cw   = trial.suggest_categorical('class_weight', ['none','balanced'])
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
                                l1_ratio=p_lr.get('l1_ratio',0.5),
                                solver='saga', class_weight=cw_v, max_iter=3000, random_state=SEED)
elif pen == 'l1':
    clf_lr = LogisticRegression(C=p_lr['C'], penalty='l1', solver='saga',
                                class_weight=cw_v, max_iter=3000, random_state=SEED)
else:
    clf_lr = LogisticRegression(C=p_lr['C'], penalty='l2', solver='saga',
                                class_weight=cw_v, max_iter=3000, random_state=SEED)
sc_lr = RobustScaler() if p_lr['scaler'] == 'robust' else StandardScaler()
model_lr = Pipeline([('sc', sc_lr), ('clf', clf_lr)])
model_lr.fit(X_train, y_train)
p_lr_test  = model_lr.predict_proba(X_test)[:, 1]
acc_lr     = accuracy_score(y_test, (p_lr_test > 0.5).astype(int))
brier_lr   = brier_score_loss(y_test, p_lr_test)
auc_lr     = roc_auc_score(y_test, p_lr_test)
print(f"  LR : acc={acc_lr:.4f}  brier={brier_lr:.4f}  AUC={auc_lr:.4f}")
print(f"  Params: {p_lr}")
gc.collect()

# ─── STEP B — XGBoost (Optuna 50 trials) ──────────────────────────────────────
print(f"\nStep B — XGBoost ({N_TRIALS} Optuna trials)...")

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

# ─── 50/50 Blend ──────────────────────────────────────────────────────────────
print("\nStep C — 50/50 LR + XGB Blend...")
p_blend    = 0.50 * p_lr_test + 0.50 * p_xgb_test
acc_blend  = accuracy_score(y_test, (p_blend > 0.5).astype(int))
brier_blend = brier_score_loss(y_test, p_blend)
auc_blend   = roc_auc_score(y_test, p_blend)

# Calibration MAE over decile bins
bins = np.linspace(0, 1, 11)
cal_mae = 0.0; n_bins = 0
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_blend >= lo) & (p_blend < hi)
    if mask.sum() >= 3:
        cal_mae += abs(p_blend[mask].mean() - y_test[mask].mean())
        n_bins  += 1
cal_mae = cal_mae / n_bins if n_bins > 0 else float('nan')

print(f"  Blend: acc={acc_blend:.4f}  brier={brier_blend:.4f}  AUC={auc_blend:.4f}  CalMAE={cal_mae:.4f}")
print(f"  Beat women's M1 ({W_M1_ACC:.4f})? {'YES ✓' if acc_blend > W_M1_ACC else 'no ✗'}")

# ─── Feature importances ──────────────────────────────────────────────────────
print("\nFeature importances — XGB (top 15):")
fi_xgb = sorted(zip(FEAT_NAMES, model_xgb.feature_importances_), key=lambda x: -x[1])
for rank, (feat, imp) in enumerate(fi_xgb[:15], 1):
    print(f"  {rank:2d}. {feat:<22}: {imp:.4f}")

print("\nFeature importances — LR |coef| (top 15):")
try:
    coefs  = abs(model_lr.named_steps['clf'].coef_[0])
    fi_lr  = sorted(zip(FEAT_NAMES, coefs), key=lambda x: -x[1])
    for rank, (feat, imp) in enumerate(fi_lr[:15], 1):
        print(f"  {rank:2d}. {feat:<22}: {imp:.4f}")
except Exception as e:
    print(f"  (failed: {e})")

# ─── ROI Simulation ───────────────────────────────────────────────────────────
print("\nROI simulation (test set 2024+)...")

test_rows  = merged[test_mask].copy().reset_index(drop=True)
test_rows['m2a_blend_prob'] = p_blend   # prob that RED wins
test_rows['y_true']         = y_test

def american_to_decimal(ml):
    ml = float(ml)
    if ml == 0 or np.isnan(ml): return None
    return 1 + 100/abs(ml) if ml < 0 else 1 + ml/100

def simulate_roi(df_sub, prob_col, stake=20):
    """Flat-stake ROI. Bets on whoever the model favors."""
    payouts = []; wins = 0; bets = 0
    for _, r in df_sub.iterrows():
        m2a_p = float(r[prob_col])
        pick_red = m2a_p > 0.5
        # Use odds from merged R_odds/B_odds
        if pick_red:
            ml  = r.get('R_odds', np.nan)
            won = int(r['y_true']) == 1
        else:
            ml  = r.get('B_odds', np.nan)
            won = int(r['y_true']) == 0
        if pd.isna(ml) or ml == 0: continue
        dec = american_to_decimal(ml)
        if dec is None: continue
        bets += 1
        if won:
            payouts.append(stake * (dec - 1))
            wins += 1
        else:
            payouts.append(-stake)
    if not payouts: return 0, 0, 0.0, 0.0
    roi = sum(payouts) / (stake * len(payouts)) * 100
    wr  = wins / bets * 100
    return bets, wins, wr, roi

def simulate_qkelly(df_sub, prob_col, bankroll=1000):
    bk = bankroll; peak = bk; wins = 0; bets = 0
    for _, r in df_sub.iterrows():
        m2a_p = float(r[prob_col])
        pick_red = m2a_p > 0.5
        p = m2a_p if pick_red else 1.0 - m2a_p
        if pick_red:
            ml = r.get('R_odds', np.nan)
            won = int(r['y_true']) == 1
        else:
            ml = r.get('B_odds', np.nan)
            won = int(r['y_true']) == 0
        if pd.isna(ml) or ml == 0: continue
        dec = american_to_decimal(ml)
        if dec is None: continue
        b = dec - 1
        q = 1 - p
        kelly = max(0.0, (b*p - q)/b) * 0.25
        bet_amt = kelly * bk
        bets += 1
        if won:
            bk += bet_amt * b; wins += 1
        else:
            bk -= bet_amt
        peak = max(peak, bk)
    roi = (bk - bankroll) / bankroll * 100
    return bets, wins/bets*100 if bets else 0, roi, bk

print(f"\n  {'Group':<15}  {'N':>4}  {'Flat WR':>8}  {'Flat ROI':>9}  {'QK ROI':>9}  {'QK End $':>9}")
print(f"  {'-'*15}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")

for at in ['ALL','CONFIRM_DOG','CONFIRM_FAV','NO_EDGE']:
    if at == 'ALL':
        sub = test_rows.copy()
    else:
        sub = test_rows[test_rows['agreement_type'] == at].copy()
    if len(sub) == 0:
        continue
    bets, wins, wr, roi = simulate_roi(sub, 'm2a_blend_prob')
    qk_bets, qk_wr, qk_roi, qk_end = simulate_qkelly(sub, 'm2a_blend_prob')
    print(f"  {at:<15}  {bets:>4}  {wr:>7.1f}%  {roi:>+8.1f}%  {qk_roi:>+8.1f}%  ${qk_end:>8,.0f}")

gc.collect()

# ─── Side-by-side comparison ──────────────────────────────────────────────────
# Women's M1 test accuracy on this filtered set
w_m1_test_preds = (merged.loc[test_mask, 'm1_prob'].values > 50).astype(int)
w_m1_test_acc   = accuracy_score(y_test, w_m1_test_preds)

print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)
print(f"\n  {'Model':<32}  {'Acc':>7}  {'Brier':>7}  {'AUC':>7}  {'CalMAE':>8}")
print(f"  {'-'*32}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
print(f"  {'Women M1 (baseline, filtered)':32}  {w_m1_test_acc:7.4f}  {'—':>7}  {'—':>7}  {'—':>8}")
print(f"  {'Women M1 full test acc':32}  {W_M1_ACC:7.4f}  {'—':>7}  {'—':>7}  {'—':>8}")
print(f"  {'Men M2A prod (reference)':32}  {MEN_M2A_ACC:7.4f}  {'—':>7}  {'—':>7}  {'—':>8}")
print(f"  {'NEW LR (method+M1)':32}  {acc_lr:7.4f}  {brier_lr:7.4f}  {auc_lr:7.4f}  {'—':>8}")
print(f"  {'NEW XGB (method+M1)':32}  {acc_xgb:7.4f}  {brier_xgb:7.4f}  {auc_xgb:7.4f}  {'—':>8}")
print(f"  {'NEW 50/50 Blend':32}  {acc_blend:7.4f}  {brier_blend:7.4f}  {auc_blend:7.4f}  {cal_mae:8.4f}")

delta_m1  = acc_blend - w_m1_test_acc
delta_men = acc_blend - MEN_M2A_ACC
s1 = '+' if delta_m1  >= 0 else ''
s2 = '+' if delta_men >= 0 else ''
print(f"\n  vs Women M1 (filtered baseline):  {s1}{delta_m1*100:.2f}pp")
print(f"  vs Men M2A prod:                  {s2}{delta_men*100:.2f}pp")

# ─── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving models...")
joblib.dump(model_lr,   'model/ufc_model2a_womens_lr.pkl')
joblib.dump(model_xgb,  'model/ufc_model2a_womens_xgb.pkl')
joblib.dump(FEAT_NAMES, 'model/feature_columns_2a_womens.pkl')
print("  Saved: model/ufc_model2a_womens_lr.pkl")
print("  Saved: model/ufc_model2a_womens_xgb.pkl")
print("  Saved: model/feature_columns_2a_womens.pkl")

meta = {
    'created':           datetime.now().isoformat(),
    'description':       'Women M2A — method odds + M1, no moneyline features',
    'features':          FEAT_NAMES,
    'n_features':        len(FEAT_NAMES),
    'blend':             '50% LR + 50% XGB',
    'train_size':        int(X_train.shape[0]),
    'test_size':         int(X_test.shape[0]),
    'temporal_split':    '2024-01-01',
    'filter':            'all 6 method odds non-null, 2018+',
    'acc_lr':            float(acc_lr),
    'acc_xgb':           float(acc_xgb),
    'acc_blend':         float(acc_blend),
    'brier_blend':       float(brier_blend),
    'auc_blend':         float(auc_blend),
    'cal_mae_blend':     float(cal_mae),
    'w_m1_acc_filtered': float(w_m1_test_acc),
    'w_m1_acc_full':     W_M1_ACC,
    'men_m2a_acc':       MEN_M2A_ACC,
    'beats_w_m1':        bool(acc_blend > w_m1_test_acc),
    'lr_params':         {k: (float(v) if isinstance(v,float) else v) for k,v in p_lr.items()},
    'xgb_params':        {k: (float(v) if isinstance(v,float) else v) for k,v in p_xgb.items()},
    'top5_features_xgb': [feat for feat, _ in fi_xgb[:5]],
}
with open('model/model2a_womens_v2_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("  Saved: model/model2a_womens_v2_metadata.json")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
