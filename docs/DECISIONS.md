# Architectural Decisions

---

## Models 3A and 3B: method prediction layer (May 2026)

**Decision:** Promoted Model 3A ("Goes the Distance") and Model 3B v2 ("Winner and Method") to production. Models are exposed via the `/method` endpoint. They run alongside M1/M2A/M2B and do not replace them.

**Model 3A — Goes the Distance (binary classifier):**
30% LR + 70% XGB, 63 features, 64.94% accuracy (+11.98pp vs 52.96% naive). Predicts whether a fight ends by decision (1) vs finish (0). Used for `goes_distance_prob` and `finish_prob` in the `/method` response. Uncalibrated — isotonic calibration on a non-chronological val slice degraded MAE from 0.0615 to 0.1895 and was discarded. A proper chronological holdout calibration pass is deferred.

Low-confidence divisions flagged: Women's Flyweight, Light Heavyweight, Bantamweight (smaller sample, higher method variance in training data). `low_confidence_division: true` is set in the response when the weight class matches.

**Model 3B v2 — Winner and Method (six-class classifier):**
40% RF + 60% XGB, 102 features, 46.56% six-class accuracy (+16.68pp vs 29.88% naive). Predicts one of six outcomes: R KO/TKO (0), R Sub (1), R Dec (2), B KO/TKO (3), B Sub (4), B Dec (5). Classes 0/3 collapse to the winner pick, giving a direction accuracy of 70.67%.

**Why M1 probability was fed into 3B:** The original 3B (v1, 99 features) reached only 67.64% direction accuracy despite having access to the same raw stats as M1 (72.81%). The gap existed because 3B had to solve winner-prediction and method-prediction simultaneously from raw features, and the joint learning task diluted the winner signal. Feeding `m1_red_win_prob` (M1's solved probability) as a feature in v2 offloaded winner-prediction to the specialist model, letting 3B focus on method-splitting. Result: direction accuracy jumped from 67.64% to 70.67% (+3.03pp), and submission recall improved from 16.0%/8.4% to 34.9%/12.1% (R/B). The M1 probability feature ranked 2nd and 3rd in XGB importance (`m1_red_win_prob_sq` at #2, `m1_red_win_prob` at #3), confirming it dominated the winner signal.

**Why 40% RF + 60% XGB:** Swept 10 blend combinations on the temporal test set. RF alone (44.96%) and XGB alone (44.74%) were close, but the 40/60 RF+XGB blend (46.56%) outperformed all LR-inclusive blends. RF contributes the most accurate leaf-level probability estimates on the method features; XGB corrects non-linear interactions. LR (41% alone) dragged accuracy down in all combinations — the six-class multinomial task has too many local non-linearities for LR's linear boundary to handle well.

**Direction accuracy gap vs M1 (2.14pp):** 3B reaches 70.67% vs M1's 72.81%. The remaining gap is structural: 3B simultaneously assigns a method to every prediction, which adds uncertainty that pure winner-prediction avoids. The gap is expected and acceptable — 3B's purpose is method-splitting, not winner-prediction.

**Accuracy:** 3A: 64.94% (+11.98pp). 3B v2: 46.56% six-class (+16.68pp), 70.67% direction (+3.03pp vs v1). Files: `ufc_model3a_lr.pkl`, `ufc_model3a_xgb.pkl`, `ufc_model3a_features.pkl`, `ufc_model3b_rf.pkl`, `ufc_model3b_xgb.pkl`, `ufc_model3b_features.pkl`.

---

## Model 2B V3: Random Forest + SPLIT floor, 20 features (May 2026)

**Decision:** Promoted Model 2B V3 (Random Forest + SPLIT probability floor 0.45) to production. All five promotion criteria passed. Previous production was a 1/3+1/3+1/3 LR+RF+XGB ensemble on 15 features.

**Why RF outperformed the ensemble on this dataset:** With a 3,007-row training universe and features that are largely derived from the same underlying signal (gap, agreement, conviction), the ensemble's diversity benefit is minimal. RF at depth≤5, min_samples_leaf=10 provided the best bias-variance tradeoff. LR and the 33/33 ensemble also passed all criteria, but RF had the highest test accuracy (71.14%) and lowest Brier (0.1900) of any passing model.

**Why SPLIT floor 0.45:** SPLIT fights (M1 and M2A disagree on the winner) have an actual historical win rate of 52.1%, but the trained model consistently predicted below 45% for these fights — a systematic underconfidence of ~7–10pp. The floor corrects this without retraining. It is applied post-prediction, before reporting win_probability. Brier improved from 0.1951 to 0.1900 with the floor applied.

**Why agreement_encoded ranked #3:** Providing an explicit ordinal (CONFIRM=3, SPLIT=2, NEAR_ZERO=1, COUNTER=0) proved more informative than asking the model to infer agreement type from the combination of m1_m2a_agree (binary) and gap_direction (±1). The ordinal encodes both the agreement state and its implied confidence ordering in a single feature.

**Why conviction_product ranked #2:** The product of M1 and M2A conviction (abs(m1_prob−0.5) × abs(m2a_prob−0.5)) captures the joint confidence of both models. High conviction_product means both models are far from coin-flip — this is the strongest predictor of the value fighter winning, beyond either model's conviction individually. It ranked above m1_m2a_agree in the binary case because it carries magnitude information not just direction.

**Why is_m1_signal was removed from frontend (retained in model):** The SPLIT + Zone≥5 + M1 conviction≥0.15 archetype showed 23.9% win rate in the training data (vs 73.3% non-signal) — a complete inversion of the 68–75% WR pattern observed in the test set. This strongly suggests the test-set pattern was a sampling artifact rather than a learnable signal. The feature is retained in the model's feature vector (RF expects 20 features) but hardcoded to 0 in the backend, and the M1 SIGNAL badge is removed from AETSlip.js.

**Why gap_signed outperforms gap_size:** gap_signed (= gap_size × gap_direction) combines magnitude and direction into one continuous feature. Correlation with outcome: +0.249 vs +0.023 for gap_size alone — 11× stronger. The direction of the gap (model more confident than Vegas vs Vegas more confident than model) is the primary signal; magnitude alone is nearly uncorrelated with outcome.

**Accuracy:** 70.51% → 71.14% (+0.63pp). Brier: 0.1943 → 0.1900 (−0.0043). COUNTER MAE: 0.1064 → 0.0180 (6× better). SPLIT MAE: 0.1239 → 0.0490 (2.5× better). Model file: `model/ufc_model2b.pkl` (RF), `model/ufc_model2b_features.pkl` (20 features), `model/ufc_model2b_config.json`.

---

## Model 1 V2: men's-only, recency weighting, QA stats, 129 features (May 2026 sprint)

**Decision:** Promoted Model 1 V2 to production. New model achieves 72.81% temporal accuracy (2024+ holdout), replacing the previous 72.08% blend.

**Why women's fights were excluded:** Women's weight classes (Strawweight, Flyweight, Bantamweight, Featherweight) were scoring 57–60% accuracy — well below the men's baseline — and pulling the blended accuracy down. The women's divisions are a structurally different prediction problem: smaller fighter pools, fewer career fights per fighter, different striking/grappling profiles, and less historical depth in the career stats dataset. Mixing them into a single model requires the model to simultaneously learn two distinct prediction tasks that share features but not patterns. Exclusion immediately rescued accuracy. A dedicated women's model is flagged as a future project.

**Why recency weighting (half-life=730 days):** The sport's meta evolves. A fight from 2015 is less informative about a fighter's current form than a fight from 2023. Exponential decay weighting (`exp(-ln(2) * days_before_cutoff / HL)`) with HL=730 days was tested against 1095d and 1460d. The 730d half-life gave the best temporal holdout accuracy and makes the model more responsive to recent fighter development. The 2025 (test) accuracy recovered from 65.9% (no weighting) to 71.0% — a substantial rescue.

**Why training window expanded to 2015:** A data quality audit of 2015–2017 fights found a maximum missing-rate delta of 10.6pp relative to 2018+ data — below the 20pp threshold set for inclusion. Expanding the window added 1,222 training rows at no accuracy cost, strengthening minority patterns in the training distribution.

**Why V2 beat V3:** V2 (recency + opponent-quality-adjusted stats + interaction features, 129 features) scored 72.81% tuned vs. V3 (recency only, 109 features) at 71.98% tuned. The QA stats (career win rate, finish rate, SLpM, SApM weighted by opponent Elo at time of each fight) contributed the largest per-feature accuracy lift. Interaction features (age × layoff, finish danger mismatch) added marginal but consistent signal. V3 tuned was also below the previous production number (72.08%), making it a regression — V2 was the only viable promotion candidate.

**New features — opponent-quality-adjusted (QA) stats:** 12 features computed as cumulative career stats where each fight's contribution is scaled by `opponent_elo / 1500` at fight time. This gives a fighter's stats weighted by the quality of competition faced — a 70% strike accuracy against elite opponents is worth more than 70% against cans. All 8 source QA metrics (win rate, finish rate, SLpM, SApM for both corners) outperformed their raw counterparts in target correlation.

**New features — interactions:** `age_x_layoff` (age × min(layoff_days, 730)), `finish_danger` (KO rate + sub rate), `got_finished_rate` (fraction of losses by finish — a chin-proxy), and `finish_danger_mismatch` (cross-multiplied finish danger vs. finish resistance between corners). Rematch features (`is_rematch`, `won_first_fight`) were tested but dropped — both fell below the |r| < 0.03 inclusion threshold.

**Women's model:** Flagged as a future project. Requires a separate career stats dataset, separate Elo ratings for women's divisions, and separate feature selection. Not a V2 scope item.

**Accuracy:** 72.08% → 72.81% (+0.73pp). Model files: `ufc_model_best.pkl` (LR pipeline), `ufc_model_xgb.pkl` (XGB, Optuna best params), `feature_columns_best.pkl` (129-feature list).

**Backend approximation for QA features:** The `/predict` route approximates QA stats at inference since fighter historical data is not available from the input payload: `qa_win_rate = career_win_rate`, `qa_finish_rate = last5_finish_rate`, `qa_SLpM = qa_SApM = 0.0`. The safety fallback (`for col in feature_columns: if col not in df_input: df_input[col] = 0`) handles any remaining gaps.

---

## Model 2: 50/50 LR+XGB blend, 42 features (May 2026 sprint)

**Decision:** Promoted new Model 2 (50% LR + 50% XGB, 42 features) to production. Previous production Model 2 was a single LR model on 23 features (72.35% accuracy). New model achieves 73.20% (+0.85pp).

**Why 50/50 blend:** Sprint swept 80/20 through 50/50 LR/XGB ratios on the test set. The 50/50 split gave the highest accuracy (73.20%), outperforming 70/30 (73.13%), 80/20 (72.67%), and single-model LR (72.67%). XGB complements LR on non-linear patterns in the odds space — in the 42-feature M2 dataset, XGB carries more signal than in M1 because the odds features are discrete enough for tree splits to help.

**Why tier_hist_win_rate matters:** This is the standout new feature (r=+0.40 with outcome, 2nd most important XGB feature). It encodes the historical win rate of fighters at each odds tier — heavy_dog (<0.30 implied prob): 17.9% historical win rate; dog (0.30–0.45): 38.2%; coinflip (0.45–0.55): 49.4%; fav (0.55–0.70): 63.7%; heavy_fav (>0.70): 82.7%. This is computed from training data and looked up at inference. It tells M2 how well-calibrated the Vegas line is at this tier, giving a prior for whether the current gap is signal or noise.

**Why split models weren't promoted:** The fav/dog split approach (separate models for when F1 is the favorite vs underdog) showed 72.45% combined accuracy vs 72.24% unified — only +0.21pp above the +0.2pp threshold. The gain is too marginal to justify the added complexity: two separate models, an f1_is_fav routing condition in the backend, and doubled maintenance surface. Unified 42-feature model at 73.20% is superior.

**Feature groups added:** 7 underdog/fav profile features + 8 method odds interaction features + 4 weight class/context features = 19 new features on top of base 23. ElasticNet regularization (l1_ratio=0.785) zeroed many of the weak new features, keeping effective model complexity low.

**Backup files:** `model/ufc_model2_best_v1_backup.pkl`, `model/ufc_model2_features_v1_backup.pkl`

**Accuracy:** 72.35% → 73.20% (+0.85pp). Model files: `ufc_model2_best.pkl` (LR), `ufc_model2_xgb.pkl` (XGB), `ufc_model2_features.pkl` (42-feature list), `model2_tier_stats.json` (tier lookup table).

---

## Blend ratio: LR 70% + XGB 30% (May 2026 optimization sprint)

**Decision:** Changed production blend from 90/10 to 70/30 (LR/XGB). No retraining — same pkl files, constants updated in `backend/main.py`.

**Why:** May 2026 optimization sprint swept ratios from 95/5 to 70/30 using the production LR and XGB models on the temporal test set (2024+). Results were non-monotonic: accuracy dipped at 85/15 and 80/20 (−0.26pp each) before recovering and jumping at 75/25 (+0.35pp) and 70/30 (+0.44pp). The dip at intermediate ratios suggests XGB's non-linear corrections are partially destructive when too weak to override LR — they introduce noise rather than signal. At 30% weight, XGB has sufficient influence to correct LR's linear misses on non-linear patterns, producing the 72.08% accuracy vs. 71.64% baseline.

**Alternatives rejected:** ElasticNet penalty, LightGBM blends, isotonic calibration, and C re-tuning all underperformed the baseline on the temporal test set. LightGBM in particular was −0.79pp to −1.23pp across all blend configurations, consistent with the dataset being too small (~4K training rows) for LGBM's tree structure.

**Previous entry (90/10):** The earlier 90/10 decision was based on experiments at the 114-feature stage where 85/15 and 80/20 showed no improvement. The 109-feature Variant A model behaves differently under XGB — fewer noisy features means XGB's predictions are cleaner and can carry more weight without degrading accuracy.

**Accuracy:** 71.64% → 72.08% (+0.44pp). No model files changed.

---

## Model architecture: LR + XGB blend (90/10)

**Decision:** Primary model is 90% LogisticRegression + 10% XGBoost, probability blend.

**Why:** LR with heavy L2 regularization (`C=0.00711`) is robust on structured tabular data with moderate row counts (~3000 training fights). XGB adds a small non-linear correction. Blending at 90/10 vs. higher XGB weights did not improve temporal accuracy — per experiments `experiment11_output.txt`, 90/10, 85/15, and 80/20 all achieved 73.24%. The 90/10 split was chosen to keep the model closer to LR's calibrated probabilities (important for Kelly sizing in Model 2).

**Alternatives considered:** Random Forest, CatBoost, pure XGB. All underperformed LR on temporal split by 1–2%. RF in particular overfit to training set patterns that don't generalize forward in time.

---

## Temporal split (train < 2024, test ≥ 2024)

**Decision:** Hard cutoff at 2024-01-01. Train on all 2018–2023 fights, test on all 2024+ fights.

**Why:** UFC fighting meta and fighter pool evolve over time. A temporal split correctly simulates the deployment scenario: the model only sees past fights when predicting future ones. Cross-validation with shuffled folds would inflate accuracy by ~3–4% due to future leakage.

**Why 2018 as train start:** Career stat features become sparse and noisy before 2018 (fewer fighters with multiple UFC fights). Restricting to 2018+ gives each fight row at least some career history context.

---

## Corner-flip augmentation

**Decision:** Training set is doubled by swapping R_/B_ columns and negating all `_dif` columns. Target flips from 1→0 and 0→1.

**Why:** The UFC arbitrarily assigns Red/Blue corners. Red corner has a slight home-field advantage in some eras, but the underlying fighter quality signal should be corner-invariant. Augmenting with flipped examples forces the model to learn relative differences, not corner assignment.

**Note:** Augmentation is applied ONLY to the training set. Test set is never augmented, so test accuracy reflects real-world prediction.

---

## Elo: K=48, base=1500, all-time

**Decision:** K=48 with all-time fight history (back to 1993).

**Why K=48:** Experiment grid over K=32,40,48,56,64,72,80,96,128 showed K=48 and K=64 achieved identical temporal accuracy (72.35%). K=48 was chosen as the production value because it was the baseline and there was no clear reason to increase K (higher K amplifies single-fight volatility, which adds noise for fighters with sparse records).

**Why all-time, not windowed:** Windowed Elo (e.g., last-50-fights only) was tested and performed 0.1–0.2% worse. All-time Elo correctly preserves legacy information for established champions.

**What Elo captures:** Relative strength accounting for opponent quality. `elo_dif` is consistently a top-5 most important feature.

---

## Shift(1) career stats — no leakage

**Decision:** All career stats are computed with `shift(1)` within each fighter group, ensuring each row only sees data from fights BEFORE the current one.

**Why critical:** Using cumulative stats that include the current fight is data leakage. For example, `cumsum().shift(0)` for `won` would include the current fight's outcome in the training feature — the model would be learning from the answer. `shift(1)` shifts the entire series so row 0 sees nothing (uses debut defaults).

**Why career_fights_updated.csv, not ufc-master.csv:** ufc-master.csv only has UFC fights. Career stats include regional/international fights, giving accurate pre-UFC win rates for fighters debuting in the UFC. This is particularly important for fighters with long regional careers (e.g., 15-1 regional record before UFC).

---

## Fight filter: R_cum_fights ≥ 1 AND B_cum_fights ≥ 1

**Decision:** Exclude fights where either fighter has zero prior UFC fights in the career dataset.

**Why:** Debut fighters have no prior career data to fill the career stats columns — all features default to neutral values (0.5 win rate, 0 finishes, etc.). Including these rows in training introduces noise because the features carry no signal. The model performs poorly on debuts by design (insufficient training signal).

**Why not ≥ 3 or higher:** Experiments showed that min=1 preserved enough training data without sacrificing accuracy. Higher thresholds reduced training set size without accuracy benefit.

---

## LR regularization: C=0.00711

**Decision:** Heavy L2 regularization, C=0.00711 (λ ≈ 141).

**Why:** The feature space has 114 features with many correlated diffs. Strong regularization prevents overfitting to training-era patterns that don't generalize. The C value was tuned via grid search on temporal accuracy and then held fixed. Lower C values (stronger regularization) marginally underperformed; higher values overfit.

---

## Model 2: odds-aware LR, 1/3 Kelly gating

**Decision:** Separate Model 2 uses opening odds as a feature. Kelly fraction = 1/3. Bets only when Model 1 probability differs from implied odds probability by ≥ 10%.

**Why separate model:** Incorporating odds into Model 1 would make it unusable for fights without odds (regional cards, early lines). Model 2 is a pure value-detection layer on top of Model 1.

**Why 1/3 Kelly:** Full Kelly criterion maximizes geometric growth but has high variance and drawdown. 1/3 Kelly is a standard risk-adjusted fraction that reduces variance substantially (≈ variance goes down by 9x) while retaining most of the growth advantage.

**Why 10% gap threshold:** Below 10%, the edge is within typical odds-line noise. Gates out marginal bets where the model is not clearly disagreeing with market consensus.

---

## Feature pruning: Variant A (109 features, May 2026)

**Decision:** Removed 5 features from the original 114-feature set:

| Feature | XGB Importance | Zero Rate | Reason |
|---------|---------------|-----------|--------|
| `title_bout_bin` | 0.0000 | 96.0% | 96% of training rows are zero; zero predictive power |
| `B_southpaw` | 0.0000 | 81.5% | Zero importance; stance is fully captured by `orth_clash` and `south_clash` |
| `B_layoff_gt365` | 0.0032 | 87.6% | Rare event (only 12% of fighters), very weak importance |
| `R_total_title_bouts` | 0.0032 | 80.3% | Very sparse, near-zero importance |
| `last10_win_rate_dif` | 0.0038 | 34.1% | Highly redundant with `career_win_rate_dif` (\|r\|=0.851) — drop the lower-correlation duplicate |

**Result:** Accuracy improved from 71.47% → 71.64% (+0.18pp) on the temporal test set (2024+).

**Why these and not others:** The audit flagged features by requiring BOTH low XGB importance (<0.5%) AND high zero rate (>70%). Features in the bottom 15 by importance that still have decent zero rates were kept — their correlation with the target (0.05–0.15) suggests they carry diffuse signal that LR picks up even when XGB doesn't. `last10_win_rate_dif` was a special case: flagged purely on redundancy grounds (correlation with `career_win_rate_dif`) rather than sparsity.

**Backup files:** `model/ufc_model_best_114_backup.pkl`, `model/ufc_model_xgb_114_backup.pkl`, `model/feature_columns_best_114_backup.pkl`

---

## is_debut flag: zero rows in career dataset (not zero UFC wins)

**Decision:** A fighter is flagged as a debut if they have NO rows in `career_fights_updated.csv` at all — not if they have zero UFC wins.

**Why:** Pre-fight `R_wins` in ufc-master.csv stores wins going INTO the fight. A fighter who won their first UFC fight has `R_wins=0` on that fight's row. Using `wins==0` as the debut flag would incorrectly flag fighters like Brando Pericic (1-0 UFC) as debuts.

**Effect in backend:** Debut fighters receive `🆕 UFC Debut` badge, a model-confidence warning, and `N/A` Kelly bet size (Model 2 not applied). Only Ben Johnston (Perth card) meets this criterion.
