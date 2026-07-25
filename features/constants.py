"""
features/constants.py — dependency-free shared constants/formulas
(8SI Phase 5 Part B).

Deliberately has NO imports from training/train_model1.py or
features/build.py: build.py imports compute_*() functions FROM
train_model1.py, so if train_model1.py also imported shared constants FROM
build.py, that would be a circular import. This module sits below both —
train_model1.py, backend/main.py, and features/build.py all import from
here, nothing imports the other way.

Moved here (from previously being duplicated in train_model1.py and
backend/main.py, with the values already matching): WC_ORDER,
WOMENS_CLASSES, layoff-bucket thresholds, and the small interaction-feature
formulas — so a definition can't silently diverge (the plan's "minimum
bar" fallback, 8si_remediation_plan.md Phase 5).
"""

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

WOMENS_CLASSES = {
    "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
}


def layoff_bucket_flags(days: float) -> dict:
    """Scalar layoff-bucket one-hot flags — for a single fighter's days-since-last-fight."""
    return {
        'lt90':    1 if days < 90 else 0,
        '90_180':  1 if 90 <= days < 180 else 0,
        '180_365': 1 if 180 <= days < 365 else 0,
        'gt365':   1 if days >= 365 else 0,
    }


def layoff_bucket_flags_vectorized(prefix: str, days_series):
    """Same thresholds as layoff_bucket_flags(), vectorized over a pandas
    Series for the trainer's bulk-training path."""
    d = days_series.fillna(180.0)
    return {
        f'{prefix}layoff_lt90':    (d < 90).astype(int),
        f'{prefix}layoff_90_180':  ((d >= 90)  & (d < 180)).astype(int),
        f'{prefix}layoff_180_365': ((d >= 180) & (d < 365)).astype(int),
        f'{prefix}layoff_gt365':   (d >= 365).astype(int),
    }


def age_x_exp(age: float, cum_fights: float) -> float:
    return age * cum_fights


def age_x_layoff(age: float, layoff_days: float, cap: float = 730) -> float:
    return age * min(layoff_days, cap)


def finish_danger(ko_finish_rate: float, sub_finish_rate: float) -> float:
    return ko_finish_rate + sub_finish_rate


def finish_danger_mismatch(r_finish_danger: float, b_finish_danger: float,
                            r_got_finished_rate: float, b_got_finished_rate: float) -> float:
    return (r_finish_danger * (1 - b_got_finished_rate)
            - b_finish_danger * (1 - r_got_finished_rate))
