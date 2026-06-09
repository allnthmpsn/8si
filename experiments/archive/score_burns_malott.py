"""Retroactive Model 2 scoring — UFC Fight Night: Burns vs Malott (April 18, 2026)."""
import sys
import math
import requests
import numpy as np
import joblib

API = 'http://localhost:8000'

card = [
    ('Mike Malott',          'Gilbert Burns',         -278,  225),
    ('Charles Jourdain',     'Kyler Phillips',        -135,  114),
    ('Mandel Nallo',         'Jai Herbert',           -180,  150),
    ('Jasmine Jasudavicius', 'Karine Silva',          -298,  240),
    ('Thiago Moises',        'Gauge Young',            140, -166),
    ('Marcio Barbosa',       'Dennis Buzukja',        -455,  350),
    ('Robert Valentin',      'Julien Leblanc',        -162,  136),
    ('Tanner Boser',         'Gokhan Saricam',         124, -148),
    ('Melissa Croden',       'Darya Zheleznyakova',   -130,  110),
    ('JJ Aldrich',           'Jamey-Lyn Horth',        130, -155),
    ('John Castaneda',       'Mark Vologdin',         -148,  124),
    ('Jamie Siraj',          'John Yannis',           -258,  210),
]

results = {
    'Mike Malott vs Gilbert Burns':          'Mike Malott',
    'Charles Jourdain vs Kyler Phillips':    'Charles Jourdain',
    'Mandel Nallo vs Jai Herbert':           'Jai Herbert',
    'Jasmine Jasudavicius vs Karine Silva':  'Jasmine Jasudavicius',
    'Thiago Moises vs Gauge Young':          'Thiago Moises',
    'Marcio Barbosa vs Dennis Buzukja':      'Marcio Barbosa',
    'Robert Valentin vs Julien Leblanc':     'Robert Valentin',
    'Tanner Boser vs Gokhan Saricam':        'Gokhan Saricam',
    'Melissa Croden vs Darya Zheleznyakova': 'Melissa Croden',
    'JJ Aldrich vs Jamey-Lyn Horth':         'JJ Aldrich',
    'John Castaneda vs Mark Vologdin':       'Draw',
    'Jamie Siraj vs John Yannis':            'Jamie Siraj',
}

model2   = joblib.load('model/ufc_model2_best.pkl')
m2_feats = joblib.load('model/ufc_model2_features.pkl')


def implied_prob(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def get_fighter(name):
    r = requests.get(f'{API}/fighter/{requests.utils.quote(name)}', timeout=10)
    if r.status_code == 200:
        return r.json()
    return None


def build_payload(f1d, f2d):
    def pf(d, prefix):
        return {
            f'{prefix}_wins':                    d.get('wins', 0),
            f'{prefix}_losses':                  d.get('losses', 0),
            f'{prefix}_total_wins':              d.get('total_wins', 0),
            f'{prefix}_total_losses':            d.get('total_losses', 0),
            f'{prefix}_Height_cms':              d.get('Height_cms', 175),
            f'{prefix}_Reach_cms':               d.get('Reach_cms', 175),
            f'{prefix}_age':                     d.get('age', 28),
            f'{prefix}_avg_SIG_STR_landed':      d.get('avg_SIG_STR_landed', 0),
            f'{prefix}_avg_SIG_STR_pct':         d.get('avg_SIG_STR_pct', 0),
            f'{prefix}_avg_TD_landed':           d.get('avg_TD_landed', 0),
            f'{prefix}_avg_TD_pct':              d.get('avg_TD_pct', 0),
            f'{prefix}_avg_SUB_ATT':             d.get('avg_SUB_ATT', 0),
            f'{prefix}_win_by_KO_TKO':           d.get('win_by_KO_TKO', 0),
            f'{prefix}_win_by_Submission':       d.get('win_by_Submission', 0),
            f'{prefix}_win_by_Decision_Unanimous': d.get('win_by_Decision_Unanimous', 0),
            f'{prefix}_win_by_Decision_Split':   d.get('win_by_Decision_Split', 0),
            f'{prefix}_win_by_Decision_Majority': d.get('win_by_Decision_Majority', 0),
            f'{prefix}_current_win_streak':      d.get('current_win_streak', 0),
            f'{prefix}_current_lose_streak':     d.get('current_lose_streak', 0),
            f'{prefix}_longest_win_streak':      d.get('longest_win_streak', 0),
            f'{prefix}_total_title_bouts':       d.get('total_title_bouts', 0),
            f'{prefix}_is_southpaw':             d.get('is_southpaw', 0),
            f'{prefix}_cum_fights':              d.get('cum_fights', 0),
            f'{prefix}_career_win_rate':         d.get('career_win_rate', 0.5),
            f'{prefix}_last5_won':               d.get('last5_won', 0.5),
            f'{prefix}_last5_finish_rate':       d.get('last5_finish_rate', 0.3),
            f'{prefix}_ko_finish_rate':          d.get('ko_finish_rate', 0),
            f'{prefix}_sub_finish_rate':         d.get('sub_finish_rate', 0),
            f'{prefix}_last3_win_rate':          d.get('last3_win_rate', 0.5),
            f'{prefix}_last10_win_rate':         d.get('last10_win_rate', 0.5),
            f'{prefix}_trend_score':             d.get('trend_score', 0),
            f'{prefix}_opp_quality':             d.get('opp_quality', 0.5),
            f'{prefix}_layoff_days':             d.get('layoff_days', 180),
            f'{prefix}_SLpM':                    d.get('SLpM', 0),
            f'{prefix}_SApM':                    d.get('SApM', 0),
            f'{prefix}_Str_Acc':                 d.get('Str_Acc', 0),
            f'{prefix}_Str_Def':                 d.get('Str_Def', 0),
            f'{prefix}_TD_Avg':                  d.get('TD_Avg', 0),
            f'{prefix}_TD_Acc':                  d.get('TD_Acc', 0),
            f'{prefix}_TD_Def':                  d.get('TD_Def', 0),
            f'{prefix}_Sub_Avg':                 d.get('Sub_Avg', 0),
            f'{prefix}_elo':                     d.get('elo', 1500),
            f'{prefix}_elo_trend':               d.get('elo_trend', 0),
            f'{prefix}_pre_ufc_wins':            0,
            f'{prefix}_pre_ufc_losses':          0,
            f'{prefix}_days_since_last':         d.get('days_since_last', 180),
            f'{prefix}_fight_frequency':         d.get('fight_frequency') or 2.0,
        }
    payload = {}
    payload.update(pf(f1d, 'F1'))
    payload.update(pf(f2d, 'F2'))
    payload['weight_class'] = 'Welterweight'
    payload['title_bout']   = False
    return payload


def get_m1_prob(f1d, f2d):
    payload = build_payload(f1d, f2d)
    r = requests.post(f'{API}/predict', json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    raw = data.get('f1_probability', data.get('fighter1_probability', 50.0))
    return raw / 100.0 if raw > 1.0 else raw


def build_m2_features(m1_prob, f1_odds, f2_odds):
    f1_raw = implied_prob(f1_odds)
    f2_raw = implied_prob(f2_odds)
    total  = f1_raw + f2_raw
    f1_nv  = f1_raw / total
    f2_nv  = f2_raw / total

    gap           = m1_prob - f1_nv
    abs_gap       = abs(gap)
    vegas_conf    = abs(f1_nv - 0.5) * 2
    f1_is_fav     = 1.0 if f1_nv > 0.5 else 0.0
    model_agrees  = 1.0 if (m1_prob > 0.5) == (f1_nv > 0.5) else 0.0
    model_conf    = abs(m1_prob - 0.5) * 2
    gap_x_vconf   = gap * vegas_conf
    joint_conf    = model_conf * vegas_conf
    gap_sq        = gap ** 2 * math.copysign(1, gap)

    # Method odds not available — zero out all 9 method features
    row = {
        'model1_prob':        m1_prob,
        'f1_no_vig':          f1_nv,
        'f2_no_vig':          f2_nv,
        'model_vs_vegas_gap': gap,
        'abs_gap':            abs_gap,
        'vegas_confidence':   vegas_conf,
        'f1_is_favorite':     f1_is_fav,
        'model_agrees':       model_agrees,
        'model_confidence':   model_conf,
        'f1_dec_implied':     0.0,
        'f1_sub_implied':     0.0,
        'f1_ko_implied':      0.0,
        'f2_dec_implied':     0.0,
        'f2_sub_implied':     0.0,
        'f2_ko_implied':      0.0,
        'dec_implied_dif':    0.0,
        'sub_implied_dif':    0.0,
        'ko_implied_dif':     0.0,
        'finish_implied':     0.0,
        'gap_x_vegas_conf':   gap_x_vconf,
        'joint_confidence':   joint_conf,
        'gap_squared':        gap_sq,
    }
    return np.array([[row[f] for f in m2_feats]]), gap, f1_nv


def kelly_size(prob_win, pick_odds, ufc_wins, bankroll=1000, max_bet=100):
    if pick_odds > 0:
        dec = pick_odds / 100 + 1
    else:
        dec = 100 / abs(pick_odds) + 1
    imp = implied_prob(pick_odds)
    k = (prob_win - imp) / (dec - 1)
    qk = max(0.0, k / 4)
    if ufc_wins == 0:
        qk /= 2
    return min(round(qk * bankroll), max_bet)


def fmt_odds(o):
    return f'+{o}' if o > 0 else str(o)


# ─── Main ────────────────────────────────────────────────────────────────────

print('=' * 40)
print('MODEL 2 — BURNS VS MALOTT CARD TEST')
print('April 18, 2026 | Retroactive Scoring')
print('Note: Method odds zeroed out (not available)')
print('=' * 40)
print()

rows         = []
not_found    = 0
value_bets   = []

for f1_name, f2_name, f1_odds, f2_odds in card:
    f1d = get_fighter(f1_name)
    f2d = get_fighter(f2_name)

    missing = []
    if f1d is None: missing.append(f1_name)
    if f2d is None: missing.append(f2_name)
    if missing:
        print(f'  WARNING: not found — {", ".join(missing)} — skipping {f1_name} vs {f2_name}')
        not_found += len(missing)
        rows.append({'f1': f1_name, 'f2': f2_name, 'skipped': True})
        continue

    m1_prob = get_m1_prob(f1d, f2d)
    X, gap, f1_nv = build_m2_features(m1_prob, f1_odds, f2_odds)

    m2_prob_f1 = model2.predict_proba(X)[0][1]

    if m2_prob_f1 > 0.5:
        pick, pick_prob, pick_odds, pick_nv = f1_name, m2_prob_f1, f1_odds, f1_nv
    else:
        pick, pick_prob, pick_odds, pick_nv = f2_name, 1 - m2_prob_f1, f2_odds, 1 - f1_nv

    pick_ufc_wins = (f1d if m2_prob_f1 > 0.5 else f2d).get('wins', 0)
    bet = kelly_size(pick_prob, pick_odds, pick_ufc_wins)
    is_value = abs(gap) > 0.07

    result_key = f'{f1_name} vs {f2_name}'
    actual_winner = results.get(result_key, '')

    rows.append({
        'f1': f1_name, 'f2': f2_name,
        'm1_pct': m1_prob * 100,
        'vegas_pct': f1_nv * 100,
        'gap': gap * 100,
        'pick': pick,
        'pick_odds': pick_odds,
        'bet': bet,
        'is_value': is_value,
        'actual': actual_winner,
        'skipped': False,
    })

    if is_value:
        value_bets.append(rows[-1])

# ─── Fight-by-fight table ────────────────────────────────────────────────────
header = f"{'Fight':<35} {'M1%':>5}  {'Vegas%':>7}  {'Gap':>6}  {'Pick':<18} {'Odds':>6}  {'Kelly':>6}  {'Value'}"
print('FIGHT-BY-FIGHT PICKS:')
print(header)
print('-' * len(header))

for r in rows:
    if r['skipped']:
        print(f"  {r['f1']} vs {r['f2']} — SKIPPED (fighter not found)")
        continue
    fight_label = f"{r['f1']} vs {r['f2']}"
    gap_str = f"{r['gap']:+.1f}%"
    value_str = '⚡' if r['is_value'] else '—'
    print(
        f"{fight_label:<35} {r['m1_pct']:>4.1f}%  {r['vegas_pct']:>6.1f}%  {gap_str:>7}  "
        f"{r['pick']:<18} {fmt_odds(r['pick_odds']):>6}  ${r['bet']:>4}  {value_str}"
    )

# ─── Value bet listing ───────────────────────────────────────────────────────
print()
print('VALUE BETS (gap > 7%):')
if not value_bets:
    print('  None flagged.')
else:
    for r in value_bets:
        print(f"  ⚡ {r['f1']} vs {r['f2']}: {r['pick']} at {fmt_odds(r['pick_odds'])} — ${r['bet']} bet")

# ─── Results scoring ─────────────────────────────────────────────────────────
print()
print('RESULTS SCORING:')
val_wins = val_losses = 0
staked = returned = 0

for r in value_bets:
    actual = r['actual']
    correct = (actual == r['pick'])

    if r['pick_odds'] > 0:
        dec = r['pick_odds'] / 100 + 1
    else:
        dec = 100 / abs(r['pick_odds']) + 1

    staked += r['bet']
    if correct:
        profit = round(r['bet'] * (dec - 1), 2)
        returned += r['bet'] + profit
        val_wins += 1
        mark = '✓'
        result_str = f'won ${profit:.0f}'
    else:
        val_losses += 1
        mark = '✗'
        result_str = f'lost ${r["bet"]}'

    print(f"  {r['pick']} {mark}  ({r['f1']} vs {r['f2']}) — {result_str}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print()
scored_rows  = [r for r in rows if not r['skipped']]
total_picks  = sum(1 for r in scored_rows if r['actual'] != 'Draw')
correct_all  = sum(1 for r in scored_rows if r['actual'] not in ('', 'Draw') and r['actual'] == r['pick'])
roi = (returned - staked) / staked * 100 if staked > 0 else 0.0

print('SUMMARY:')
print(f"  Total fights:          {len(card)}")
print(f"  Fighters not found:    {not_found}")
print(f"  Value bets flagged:    {len(value_bets)} / {len(scored_rows)}")
print(f"  Value bet record:      {val_wins}-{val_losses}")
print(f"  Value bet staked:      ${staked}")
print(f"  Value bet returned:    ${returned:.0f}")
print(f"  Value bet ROI:         {roi:+.1f}%")
print(f"  Overall pick accuracy: {correct_all}/{total_picks} ({correct_all/total_picks*100:.1f}%)" if total_picks else "  Overall pick accuracy: N/A")
print()
print('  vs Sterling card:      9-10 value bets, +11.80% ROI (holdout avg)')
print(f"  Burns card:            {len(value_bets)} value bets, {roi:+.1f}% ROI")
print('=' * 40)
