#!/usr/bin/env python3
"""
training/ratings.py — 8SI v2 Stage 3.3, Glicko-2 rating system

Standard Glicko-2 (Glickman, "Example of the Glicko-2 system", 2013),
alongside — not replacing — training/train_model1.py's own Elo. Unlike
Elo, Glicko-2 tracks a per-fighter RATING DEVIATION (RD, uncertainty)
and VOLATILITY, both of which shrink as a fighter accumulates fights and
grow (RD only) with inactivity. Two intended uses (8SI v2 Stage 3.3):
  - rd_max = max(R_rd, B_rd) as a MODEL FEATURE — ship only if it
    improves walk-forward log loss (see experiments/glicko_v2/).
  - rd_max as a BET GATE (Stage 4) — ships regardless, since gating on
    "how uncertain are we about either fighter" needs no training, just
    a threshold on an already-computed number.

One-fight-per-period, processed in a single chronological pass (same
architecture as compute_elo()) — each fight IS its own rating period for
both fighters involved, not batched. Does NOT separately model
inter-fight calendar-time decay (the "no games in period t" RD-inflation
step in the full Glicko-2 spec) — Elo already has its own explicit,
tested layoff-regression feature for that; adding a second, differently-
shaped inactivity model here would be a second, hard-to-reconcile
mechanism for the same underlying idea. Scoping choice, not an
oversight.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GLICKO_SCALE = 173.7178
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
TAU = 0.5  # system constant, constrains volatility change — Glickman's own suggested range 0.3-1.2


def _g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)


def _E(mu, mu_j, phi_j):
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _new_volatility(phi, delta, v, sigma, tau=TAU):
    """Illinois algorithm (regula falsi variant) per Glickman's spec section 5, step 5."""
    a = math.log(sigma ** 2)
    eps = 1e-6

    def f(x):
        ex = math.exp(x)
        num = ex * (delta ** 2 - phi ** 2 - v - ex)
        den = 2.0 * (phi ** 2 + v + ex) ** 2
        return num / den - (x - a) / tau ** 2

    A = a
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    fA, fB = f(A), f(B)
    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC
    return math.exp(A / 2.0)


def _update_one(mu, phi, sigma, mu_j, phi_j, s):
    """One opponent, one game (a single UFC fight) — closed-form n=1 case
    of Glickman's general multi-opponent update."""
    g_j = _g(phi_j)
    E_j = _E(mu, mu_j, phi_j)
    v = 1.0 / (g_j ** 2 * E_j * (1.0 - E_j))
    delta = v * g_j * (s - E_j)

    sigma_new = _new_volatility(phi, delta, v, sigma)
    phi_star = math.sqrt(phi ** 2 + sigma_new ** 2)
    phi_new = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    mu_new = mu + phi_new ** 2 * g_j * (s - E_j)
    return mu_new, phi_new, sigma_new


def compute_glicko2(df_all):
    """Same input contract as compute_elo(): df_all has R_fighter,
    B_fighter, date, Winner ('Red'/'Blue'/other). Returns history_df
    [fighter, opponent, date, rating_before, rd_before, rating_after,
    rd_after] — rating_before/rd_before is what a caller merge_asof's
    onto a target fight (pre-fight state, no leakage, same convention as
    compute_elo()'s elo_before)."""
    df_sorted = df_all.sort_values('date').reset_index(drop=True)
    default_state = (0.0, DEFAULT_RD / GLICKO_SCALE, DEFAULT_VOL)  # (mu, phi, sigma) — mu=0 <=> rating=DEFAULT_RATING
    state = {}  # fighter -> (mu, phi, sigma) in Glicko-2 internal scale
    history_rows = []

    for _, row in df_sorted.iterrows():
        r, b = row['R_fighter'], row['B_fighter']
        winner = row['Winner']

        mu_r, phi_r, sigma_r = state.get(r, default_state)
        mu_b, phi_b, sigma_b = state.get(b, default_state)

        # Same convention as compute_elo(): a decisive win/loss gets full
        # credit, anything else (Draw/No Contest — 8/7183 fights) gets
        # 0.5/0.5 partial credit rather than being skipped, so both
        # rating systems treat non-decisive results identically instead
        # of silently diverging on 0.1% of fights.
        if winner == 'Red':
            s_r, s_b = 1.0, 0.0
        elif winner == 'Blue':
            s_r, s_b = 0.0, 1.0
        else:
            s_r, s_b = 0.5, 0.5

        rating_r_before = mu_r * GLICKO_SCALE + DEFAULT_RATING
        rd_r_before = phi_r * GLICKO_SCALE
        rating_b_before = mu_b * GLICKO_SCALE + DEFAULT_RATING
        rd_b_before = phi_b * GLICKO_SCALE

        mu_r_new, phi_r_new, sigma_r_new = _update_one(mu_r, phi_r, sigma_r, mu_b, phi_b, s_r)
        mu_b_new, phi_b_new, sigma_b_new = _update_one(mu_b, phi_b, sigma_b, mu_r, phi_r, s_b)

        state[r] = (mu_r_new, phi_r_new, sigma_r_new)
        state[b] = (mu_b_new, phi_b_new, sigma_b_new)

        history_rows.append({
            'fighter': r, 'opponent': b, 'date': row['date'],
            'rating_before': rating_r_before, 'rd_before': rd_r_before,
            'rating_after': mu_r_new * GLICKO_SCALE + DEFAULT_RATING, 'rd_after': phi_r_new * GLICKO_SCALE,
        })
        history_rows.append({
            'fighter': b, 'opponent': r, 'date': row['date'],
            'rating_before': rating_b_before, 'rd_before': rd_b_before,
            'rating_after': mu_b_new * GLICKO_SCALE + DEFAULT_RATING, 'rd_after': phi_b_new * GLICKO_SCALE,
        })

    hist = pd.DataFrame(history_rows).sort_values(['fighter', 'date']).reset_index(drop=True)
    return hist


if __name__ == '__main__':
    from training.train_model1 import DATA
    master = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    master['date'] = pd.to_datetime(master['date'])
    hist = compute_glicko2(master)
    print(f'Rows: {len(hist):,}  |  Fighters: {hist["fighter"].nunique():,}')
    latest = hist.sort_values('date').groupby('fighter').tail(1)
    print(latest.sort_values('rating_after', ascending=False)[['fighter', 'rating_after', 'rd_after']].head(10).to_string())
