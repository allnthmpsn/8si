# Model 1 V2 Sprint — Findings
_Generated: 2026-05-11 16:36_

---

## Summary

| Metric | Value |
|--------|-------|
| Production baseline (all fights, 70/30) | 72.08% |
| Men's-only baseline (109 features, 70/30) | 69.27% |
| Best variant | V1 |
| Best variant accuracy | 71.56% |
| Delta vs men's baseline | +2.29pp |
| Delta vs production | -0.52pp |

---

## SETUP — Men's Only Filter

- Women's fights removed: **695**
- Debut-filtered fights removed: 12
- Train rows (pre-aug): 2,407
- Test rows: 960
- Men's baseline accuracy: **69.27%**

---

## STEP 1 — Recency Weighting

| Half-life | Accuracy | Delta vs baseline |
|-----------|----------|-------------------|
| 730d (2yr) | 71.56% | +2.29pp ← BEST |
| 1095d (3yr) | 70.31% | +1.04pp |
| 1460d (4yr) | 70.42% | +1.15pp |

**Best half-life:** 730 days
**Best accuracy:** 71.56%

---

## STEP 2 — Opponent Quality Adjusted Stats

| Feature | Raw r | QA r | Better? |
|---------|-------|------|---------|
| R_qa_win_rate | +0.1470 | +0.1489 | ✓ QA |
| R_qa_finish_rate | +0.1019 | +0.1063 | ✓ QA |
| R_qa_SLpM | +0.0981 | +0.1460 | ✓ QA |
| R_qa_SApM | -0.1235 | -0.1502 | ✓ QA |
| qa_win_rate_dif | +0.2031 | +0.2048 | ✓ QA |
| qa_finish_rate_dif | +0.1186 | +0.1215 | ✓ QA |
| qa_SLpM_dif | +0.1777 | +0.2044 | ✓ QA |
| qa_SApM_dif | -0.1885 | -0.2044 | ✓ QA |

QA features outperform raw on **8/8** metrics.

---

## STEP 3 — New Interaction Features

**Kept** (|r| ≥ 0.03): ['R_age_x_layoff', 'B_age_x_layoff', 'age_x_layoff_dif', 'R_finish_danger', 'B_finish_danger', 'finish_danger_mismatch', 'R_got_finished_rate', 'B_got_finished_rate']

**Dropped** (|r| < 0.03): ['is_rematch', 'won_first_fight']

---

## STEP 4 — Training Window Expansion

- Max missing-rate delta (2015-17 vs 2018+): **10.6pp**
- Threshold: 20pp
- Decision: **Include 2015-2017**
- Expanded accuracy: 71.56%

---

## STEP 5 — Variant Results

| Variant | Features | Accuracy | Delta vs baseline |
|---------|----------|----------|-------------------|
| V1 | 109 | 71.56% | +2.29pp |
| V2 | 129 | 71.46% | +2.19pp |
| V3 | 109 | 71.56% | +2.29pp |

---

## Recommendation

**Recommended variant: V1** — 71.56% temporal accuracy (+2.29pp vs men's baseline, -0.52pp vs production 72.08%).

**Do not promote to production until reviewed.**

### Promotion checklist
- [ ] Review per-year accuracy for regressions in any single year
- [ ] Confirm backend can accept men's-only filter at inference or confirm
      that the model handles women's fights gracefully (it was not trained on them)
- [ ] Update model_metadata.json with men's-only flag if promoting
- [ ] A/B test on upcoming card before full promotion
