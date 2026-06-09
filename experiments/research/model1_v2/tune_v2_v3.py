#!/usr/bin/env python3
"""
Task 1 & 1B — Optuna tuning for Variant V2 (129 features) and V3 (109 features).
Both use: men's only, expanded window 2015+, recency HL=730d, 70/30 LR/XGB blend.
V2 adds QA stats + interaction features on top of 109.
Max 25 Optuna trials, 5-fold CV on training set, final eval on 2024+ temporal test.

Run from project root:
  python experiments/research/model1_v2/tune_v2_v3.py
"""
import sys, os, warnings, gc, json, traceback
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Import sprint helpers ────────────────────────────────────────────────────
SPRINT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(SPRINT_DIR)))
DATA       = os.path.join(ROOT, 'data')
OUT        = SPRINT_DIR

sys.path.insert(0, SPRINT_DIR)
from sprint import (
    build_master, compute_weights, corner_flip,
    compute_career_stats, compute_elo,
    FEAT_109, WOMENS_CLASSES, LR_WEIGHT, XGB_WEIGHT,
)

TRAIN_CUTOFF = '2024-01-01'
HL_DAYS      = 730
N_TRIALS     = 25
CV_FOLDS     = 5
LR_C         = 0.00711


def div(s): print(f'\n{"─"*62}'); print(f'  {s}'); print(f'{"─"*62}')
def ok(s):  print(f'  ✓ {s}')
def info(s):print(f'  {s}')


def per_year(df_te, y_pred_arr):
    df_te = df_te.copy()
    df_te['_pred'] = y_pred_arr
    for yr, grp in df_te.groupby(df_te['date'].dt.year):
        ya = accuracy_score(grp['target'], grp['_pred'])
        print(f'    {yr}: {ya:.3f}  ({len(grp):,} fights)')


# ─────────────────────────────────────────────────────────────────────────────
# Build V2 dataset: 2015+, men's, QA stats, interaction features
# ─────────────────────────────────────────────────────────────────────────────
def build_v2_dataset():
    info('Building V2 dataset (2015+, men\'s only, QA + interaction features)...')

    from collections import defaultdict

    df, n_rem, n_filt, career_df_raw, elo_hist = build_master(
        date_from='2015-01-01', mens_only=True)

    info(f'  Base dataset: {len(df):,} rows (after debut filter)')

    # ── Opponent elo lookup ──────────────────────────────────────────────────
    elo_for_merge = elo_hist[['fighter', 'date', 'elo_before']].copy()

    career_df_raw2 = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'))
    career_df_raw2['date'] = pd.to_datetime(career_df_raw2['date'])
    career_df_raw2 = career_df_raw2.sort_values(['fighter', 'date']).reset_index(drop=True)

    opp_elo_df = career_df_raw2[['fighter', 'opponent', 'date', 'won', 'got_finish']].copy()
    opp_elo_df = opp_elo_df.rename(columns={'opponent': 'opp_name'})
    opp_elo_lookup = elo_for_merge.rename(
        columns={'fighter': 'opp_name', 'elo_before': 'opp_elo_before'})
    opp_elo_df = pd.merge_asof(
        opp_elo_df.sort_values('date'),
        opp_elo_lookup.sort_values('date'),
        on='date', by='opp_name', direction='backward'
    )
    opp_elo_df['opp_elo_before'] = opp_elo_df['opp_elo_before'].fillna(1500.0)
    opp_elo_df['elo_weight']     = opp_elo_df['opp_elo_before'] / 1500.0

    # ── QA stats ─────────────────────────────────────────────────────────────
    def _qa_stats(group):
        group = group.sort_values('date').copy()
        n = len(group)
        qa_wr = np.full(n, 0.5)
        qa_fr = np.full(n, 0.0)
        qa_sl = np.full(n, 0.0)
        qa_sa = np.full(n, 0.0)
        cum_ew  = 0.0
        cum_eww = 0.0
        cum_ewf = 0.0
        cum_fights = 0
        cum_off = 0.0
        cum_def = 0.0
        for i in range(n):
            if cum_ew > 0:
                qa_wr[i] = cum_eww / cum_ew
                qa_fr[i] = cum_ewf / cum_ew
            if cum_fights > 0:
                qa_sl[i] = cum_off / cum_fights
                qa_sa[i] = cum_def / cum_fights
            ew  = group.iloc[i]['elo_weight']
            w   = group.iloc[i]['won']
            f   = group.iloc[i]['got_finish'] if pd.notna(group.iloc[i]['got_finish']) else 0.0
            cum_ew  += ew
            cum_eww += ew * w
            cum_ewf += ew * f
            cum_off += ew * w
            cum_def += ew * (1.0 - w)
            cum_fights += 1
        return pd.DataFrame({
            'fighter':        group['fighter'].values,
            'date':           group['date'].values,
            'qa_win_rate':    qa_wr,
            'qa_finish_rate': qa_fr,
            'qa_SLpM':        qa_sl,
            'qa_SApM':        qa_sa,
        })

    qa_list = []
    for fighter, grp in opp_elo_df.groupby('fighter', sort=False):
        qa_list.append(_qa_stats(grp))
    qa_df = pd.concat(qa_list, ignore_index=True).sort_values(['fighter', 'date'])
    del qa_list; gc.collect()

    qa_r = qa_df.rename(columns={'fighter': 'R_fighter',
        'qa_win_rate': 'R_qa_win_rate', 'qa_finish_rate': 'R_qa_finish_rate',
        'qa_SLpM': 'R_qa_SLpM', 'qa_SApM': 'R_qa_SApM'})
    qa_b = qa_df.rename(columns={'fighter': 'B_fighter',
        'qa_win_rate': 'B_qa_win_rate', 'qa_finish_rate': 'B_qa_finish_rate',
        'qa_SLpM': 'B_qa_SLpM', 'qa_SApM': 'B_qa_SApM'})

    df = pd.merge_asof(df.sort_values('date'), qa_r.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), qa_b.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')

    for c in ['R_qa_win_rate', 'R_qa_finish_rate', 'R_qa_SLpM', 'R_qa_SApM',
              'B_qa_win_rate', 'B_qa_finish_rate', 'B_qa_SLpM', 'B_qa_SApM']:
        fill_val = 0.5 if 'win_rate' in c else 0.0
        df[c] = df[c].fillna(fill_val)

    df['qa_win_rate_dif']    = df['R_qa_win_rate']    - df['B_qa_win_rate']
    df['qa_finish_rate_dif'] = df['R_qa_finish_rate'] - df['B_qa_finish_rate']
    df['qa_SLpM_dif']        = df['R_qa_SLpM']        - df['B_qa_SLpM']
    df['qa_SApM_dif']        = df['R_qa_SApM']        - df['B_qa_SApM']
    del qa_df, qa_r, qa_b; gc.collect()

    # ── Interaction features ──────────────────────────────────────────────────
    for p in ['R', 'B']:
        df[f'{p}_age_x_layoff'] = (
            pd.to_numeric(df[f'{p}_age'], errors='coerce') *
            df[f'{p}_layoff_days'].clip(upper=730)
        )
    df['age_x_layoff_dif'] = df['R_age_x_layoff'] - df['B_age_x_layoff']

    # got_finished_rate from career_fights_updated
    cdf = career_df_raw2.sort_values(['fighter', 'date']).copy()
    cdf['is_loss']    = (cdf['won'] == 0).astype(float)
    cdf['is_fin_loss'] = ((cdf['won'] == 0) & (cdf['got_finish'].fillna(0) == 1)).astype(float)
    g = cdf.groupby('fighter', sort=False)
    cdf['_cs_losses']   = g['is_loss'].cumsum()    - cdf['is_loss']
    cdf['_cs_fin_loss'] = g['is_fin_loss'].cumsum() - cdf['is_fin_loss']
    cdf['got_finished_rate'] = np.where(
        cdf['_cs_losses'] > 0,
        cdf['_cs_fin_loss'] / cdf['_cs_losses'], 0.5)
    chin_df = cdf[['fighter', 'date', 'got_finished_rate']].sort_values(['fighter', 'date'])
    chin_r = chin_df.rename(columns={'fighter': 'R_fighter', 'got_finished_rate': 'R_got_finished_rate'})
    chin_b = chin_df.rename(columns={'fighter': 'B_fighter', 'got_finished_rate': 'B_got_finished_rate'})
    df = pd.merge_asof(df.sort_values('date'), chin_r.sort_values('date'),
                       on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), chin_b.sort_values('date'),
                       on='date', by='B_fighter', direction='backward')
    df['R_got_finished_rate'] = df['R_got_finished_rate'].fillna(0.5)
    df['B_got_finished_rate'] = df['B_got_finished_rate'].fillna(0.5)
    del cdf, chin_df, chin_r, chin_b; gc.collect()

    for p in ['R', 'B']:
        df[f'{p}_finish_danger'] = df[f'{p}_ko_finish_rate'] + df[f'{p}_sub_finish_rate']
        df[f'{p}_finish_resistance'] = 1.0 - df[f'{p}_got_finished_rate']
    df['finish_danger_mismatch'] = (
        df['R_finish_danger'] * df['B_finish_resistance'] -
        df['B_finish_danger'] * df['R_finish_resistance']
    )

    # ── Final feature list ────────────────────────────────────────────────────
    QA_FEATS = [
        'R_qa_win_rate', 'R_qa_finish_rate', 'R_qa_SLpM', 'R_qa_SApM',
        'B_qa_win_rate', 'B_qa_finish_rate', 'B_qa_SLpM', 'B_qa_SApM',
        'qa_win_rate_dif', 'qa_finish_rate_dif', 'qa_SLpM_dif', 'qa_SApM_dif',
    ]
    INT_FEATS = [
        'R_age_x_layoff', 'B_age_x_layoff', 'age_x_layoff_dif',
        'R_finish_danger', 'B_finish_danger', 'finish_danger_mismatch',
        'R_got_finished_rate', 'B_got_finished_rate',
    ]
    FEAT_V2 = FEAT_109 + QA_FEATS + INT_FEATS

    for col in FEAT_V2:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df = df.sort_values('date').reset_index(drop=True)
    info(f'  V2 feature count: {len(FEAT_V2)}')
    gc.collect()
    return df, FEAT_V2


# ─────────────────────────────────────────────────────────────────────────────
# Build V3 dataset: 2015+, men's, 109 features only
# ─────────────────────────────────────────────────────────────────────────────
def build_v3_dataset():
    info('Building V3 dataset (2015+, men\'s only, 109 features)...')
    df, n_rem, n_filt, _, _ = build_master(date_from='2015-01-01', mens_only=True)
    for col in FEAT_109:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df = df.sort_values('date').reset_index(drop=True)
    info(f'  V3 rows: {len(df):,}')
    gc.collect()
    return df, FEAT_109


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning function
# ─────────────────────────────────────────────────────────────────────────────
def tune_variant(df, feat_list, variant_name):
    """
    Run 25 Optuna trials optimizing 5-fold CV accuracy on training set.
    Returns best params and evaluates on 2024+ temporal test.
    """
    train_mask = df['date'] < TRAIN_CUTOFF
    test_mask  = df['date'] >= TRAIN_CUTOFF

    X_tr_raw = df.loc[train_mask, feat_list].reset_index(drop=True)
    y_tr_raw = df.loc[train_mask, 'target'].reset_index(drop=True)
    d_tr_raw = df.loc[train_mask, 'date'].reset_index(drop=True)
    X_te     = df.loc[test_mask,  feat_list].reset_index(drop=True)
    y_te     = df.loc[test_mask,  'target'].reset_index(drop=True)
    df_te    = df[test_mask].copy().reset_index(drop=True)

    # Recency weights for training
    w_raw = pd.Series(compute_weights(d_tr_raw, half_life_days=HL_DAYS), index=y_tr_raw.index)

    # Corner-flip augmented training set + weights
    X_tr_aug, y_tr_aug, w_tr_aug = corner_flip(X_tr_raw, y_tr_raw, w_raw)

    info(f'  Train (pre-aug): {len(X_tr_raw):,} | Train (post-aug): {len(X_tr_aug):,} | Test: {len(X_te):,}')

    # Fixed LR for blending
    model_lr_fixed = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=LR_C, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    w_arr = w_tr_aug.values if hasattr(w_tr_aug, 'values') else w_tr_aug
    model_lr_fixed.fit(X_tr_aug, y_tr_aug, lr__sample_weight=w_arr)

    # 5-fold CV objective
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_categorical('n_estimators', [100, 200, 300, 400]),
            'learning_rate':    trial.suggest_categorical('learning_rate', [0.01, 0.03, 0.05, 0.1]),
            'max_depth':        trial.suggest_categorical('max_depth', [3, 4, 5, 6]),
            'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 3, 5, 10]),
            'subsample':        trial.suggest_categorical('subsample', [0.7, 0.8, 0.9]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.8]),
            'gamma':            trial.suggest_categorical('gamma', [0, 0.1, 0.3]),
            'reg_alpha':        trial.suggest_categorical('reg_alpha', [0, 0.1, 0.5]),
            'reg_lambda':       trial.suggest_categorical('reg_lambda', [0.5, 1.0, 2.0]),
            'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1,
        }
        cv_accs = []
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_tr_aug, y_tr_aug)):
            X_fold_tr = X_tr_aug.iloc[tr_idx]
            y_fold_tr = y_tr_aug.iloc[tr_idx]
            w_fold_tr = w_arr[tr_idx]
            X_fold_va = X_tr_aug.iloc[va_idx]
            y_fold_va = y_tr_aug.iloc[va_idx]

            xgb = XGBClassifier(**params)
            xgb.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)

            p_lr  = model_lr_fixed.predict_proba(X_fold_va)
            p_xgb = xgb.predict_proba(X_fold_va)
            p     = LR_WEIGHT * p_lr + XGB_WEIGHT * p_xgb
            y_pred = (p[:, 1] > 0.5).astype(int)
            cv_accs.append(accuracy_score(y_fold_va, y_pred))

        return float(np.mean(cv_accs))

    info(f'  Running {N_TRIALS} Optuna trials (5-fold CV)...')
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    gc.collect()

    best_params = study.best_params
    best_cv_acc = study.best_value
    info(f'  Best CV accuracy: {best_cv_acc:.4f} ({best_cv_acc*100:.2f}%)')

    # Train final model on full augmented training set with best params
    best_xgb_params = {**best_params,
                       'random_state': 42, 'eval_metric': 'logloss',
                       'verbosity': 0, 'n_jobs': 1}
    model_xgb_final = XGBClassifier(**best_xgb_params)
    model_xgb_final.fit(X_tr_aug, y_tr_aug, sample_weight=w_arr)
    gc.collect()

    # Temporal test evaluation
    p_lr  = model_lr_fixed.predict_proba(X_te)
    p_xgb = model_xgb_final.predict_proba(X_te)
    p     = LR_WEIGHT * p_lr + XGB_WEIGHT * p_xgb
    y_pred = (p[:, 1] > 0.5).astype(int)
    test_acc = accuracy_score(y_te, y_pred)

    info(f'  Temporal test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')

    cv_diverge = abs(best_cv_acc - test_acc) > 0.03
    if cv_diverge:
        info(f'  ⚠ CV vs test divergence: {abs(best_cv_acc - test_acc)*100:.2f}pp '
             f'— possible overfit to CV splits')
    else:
        info(f'  ✓ CV vs test gap: {abs(best_cv_acc - test_acc)*100:.2f}pp (within 3pp)')

    info(f'\n  Per-year breakdown:')
    per_year(df_te, y_pred)

    # Top 15 XGB importance
    feat_imp = sorted(zip(feat_list, model_xgb_final.feature_importances_),
                      key=lambda x: x[1], reverse=True)[:15]
    info(f'\n  Top 15 XGB features ({variant_name} tuned):')
    for feat, imp in feat_imp:
        info(f'    {feat:<38} {imp:.4f}')

    # Save
    lr_path  = os.path.join(OUT, f'variant_{variant_name}_tuned_lr.pkl')
    xgb_path = os.path.join(OUT, f'variant_{variant_name}_tuned_xgb.pkl')
    joblib.dump(model_lr_fixed,    lr_path)
    joblib.dump(model_xgb_final,   xgb_path)
    ok(f'Saved {variant_name} tuned models')

    return {
        'variant': variant_name,
        'n_features': len(feat_list),
        'best_params': best_params,
        'cv_acc': best_cv_acc,
        'test_acc': test_acc,
        'cv_diverge': cv_diverge,
        'top15': [f for f, _ in feat_imp],
        'lr_path': lr_path,
        'xgb_path': xgb_path,
        'feat_list': feat_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decision rule
# ─────────────────────────────────────────────────────────────────────────────
def decide_winner(v2_result, v3_result, untuned_best=0.715625):
    print(f'\n{"=" * 62}')
    print('  DECISION')
    print(f'{"=" * 62}')
    print(f'  Untuned V1/V3 baseline:    {untuned_best*100:.2f}%')
    print(f'  V2 tuned temporal test:    {v2_result["test_acc"]*100:.2f}%')
    print(f'  V3 tuned temporal test:    {v3_result["test_acc"]*100:.2f}%')

    v2_acc = v2_result['test_acc']
    v3_acc = v3_result['test_acc']

    if v2_acc > v3_acc and v2_acc > untuned_best:
        winner = 'V2'
        reason = f'V2 tuned ({v2_acc*100:.2f}%) beats V3 ({v3_acc*100:.2f}%) and baseline ({untuned_best*100:.2f}%)'
    elif v3_acc >= v2_acc and v3_acc > untuned_best:
        winner = 'V3'
        reason = f'V3 tuned ({v3_acc*100:.2f}%) beats V2 ({v2_acc*100:.2f}%) and baseline — prefer simpler model'
    elif v2_acc > v3_acc:
        winner = 'V2'
        reason = (f'Neither beats baseline — V2 ({v2_acc*100:.2f}%) > V3 ({v3_acc*100:.2f}%) — '
                  f'promote higher scorer')
    else:
        winner = 'V3'
        reason = (f'Neither beats baseline — V3 ({v3_acc*100:.2f}%) ≥ V2 ({v2_acc*100:.2f}%) — '
                  f'prefer fewer features (109 vs {v2_result["n_features"]})')

    print(f'\n  → WINNER: {winner}')
    print(f'    Reason: {reason}')
    return winner, v2_result if winner == 'V2' else v3_result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('=' * 62)
    print('  Task 1 & 1B — Optuna Tuning: V2 (129f) and V3 (109f)')
    print(f'  Trials: {N_TRIALS} | CV folds: {CV_FOLDS} | HL: {HL_DAYS}d')
    print(f'  Blend: {int(LR_WEIGHT*100)}% LR + {int(XGB_WEIGHT*100)}% XGB')
    print('=' * 62)

    # ── TASK 1: Tune V2 ─────────────────────────────────────────────────────
    div('TASK 1 — V2 Optuna Tuning (129 features, 2015+, QA + interaction)')
    try:
        df_v2, feat_v2 = build_v2_dataset()
        v2_result = tune_variant(df_v2, feat_v2, 'V2')
        del df_v2; gc.collect()
        print(f'\n  ── TASK 1 DONE ──')
        print(f'  V2 tuned: CV={v2_result["cv_acc"]*100:.2f}%  '
              f'Test={v2_result["test_acc"]*100:.2f}%  '
              f'(untuned was 71.46%)')
    except Exception:
        print('\n  *** TASK 1 FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    # ── TASK 1B: Tune V3 ────────────────────────────────────────────────────
    div('TASK 1B — V3 Optuna Tuning (109 features, 2015+, recency only)')
    try:
        df_v3, feat_v3 = build_v3_dataset()
        v3_result = tune_variant(df_v3, feat_v3, 'V3')
        del df_v3; gc.collect()
        print(f'\n  ── TASK 1B DONE ──')
        print(f'  V3 tuned: CV={v3_result["cv_acc"]*100:.2f}%  '
              f'Test={v3_result["test_acc"]*100:.2f}%  '
              f'(untuned was 71.56%)')
    except Exception:
        print('\n  *** TASK 1B FAILED ***')
        traceback.print_exc()
        sys.exit(1)

    # ── Decision ─────────────────────────────────────────────────────────────
    winner_name, winner_result = decide_winner(v2_result, v3_result)

    # Save tuning results JSON for Task 2
    tuning_out = {
        'V2': {k: v for k, v in v2_result.items() if k not in ('lr_path', 'xgb_path', 'feat_list')},
        'V3': {k: v for k, v in v3_result.items() if k not in ('lr_path', 'xgb_path', 'feat_list')},
        'winner': winner_name,
        'winner_test_acc': winner_result['test_acc'],
        'winner_n_features': winner_result['n_features'],
        'untuned_baseline': 0.715625,
    }
    with open(os.path.join(OUT, 'tuning_results.json'), 'w') as fh:
        json.dump(tuning_out, fh, indent=2, default=str)
    ok('Tuning results saved to tuning_results.json')

    print(f'\n{"=" * 62}')
    print(f'  TUNING COMPLETE')
    print(f'  Winner for Task 2 promotion: {winner_name}')
    print(f'  Accuracy: {winner_result["test_acc"]*100:.2f}%  '
          f'({len(winner_result["feat_list"])} features)')
    print(f'{"=" * 62}\n')

    return winner_name, winner_result, v2_result, v3_result


if __name__ == '__main__':
    main()
