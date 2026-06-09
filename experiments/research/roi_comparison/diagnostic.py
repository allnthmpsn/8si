import pandas as pd
import numpy as np

df = pd.read_csv('data/value_bet_log.csv')

print("=== COLUMN INSPECTION ===")
print(df.columns.tolist())
print()

# Print first 10 rows of the key columns
key_cols = ['m1_prob', 'm2a_prob', 'gap_direction', 'gap_size',
            'value_bet_won', 'closing_odds', 'm1_m2a_agree']
available = [c for c in key_cols if c in df.columns]
# also grab pick_won since that's the actual column name
if 'pick_won' in df.columns:
    available_plus = available + ['pick_won']
else:
    available_plus = available
print(df[[c for c in available_plus if c in df.columns]].head(10).to_string())
print()

# Use pick_won as fallback if value_bet_won missing
won_col = 'value_bet_won' if 'value_bet_won' in df.columns else 'pick_won'
print(f"Using outcome column: '{won_col}'")
print()

print("=== m1_prob distribution ===")
print(f"m1_prob mean: {df['m1_prob'].mean():.4f}")
print(f"m1_prob > 0.5: {(df['m1_prob'] > 0.5).mean():.4f}")
print(f"m1_prob < 0.5: {(df['m1_prob'] < 0.5).mean():.4f}")
print()

pos_gap = df[df['gap_direction'] == 1]
neg_gap = df[df['gap_direction'] == -1]

print("=== m1_prob when gap_direction = 1 (value fighter IS model pick) ===")
print(f"m1_prob mean: {pos_gap['m1_prob'].mean():.4f}")
print(f"m1_prob > 0.5: {(pos_gap['m1_prob'] > 0.5).mean():.4f}")
print()

print("=== m1_prob when gap_direction = -1 (value fighter NOT model pick) ===")
print(f"m1_prob mean: {neg_gap['m1_prob'].mean():.4f}")
print(f"m1_prob > 0.5: {(neg_gap['m1_prob'] > 0.5).mean():.4f}")
print()

print("=== outcome column interpretation ===")
print(f"{won_col} mean (overall):        {df[won_col].mean():.4f}")
print(f"When gap_direction= 1:          {pos_gap[won_col].mean():.4f}")
print(f"When gap_direction=-1:          {neg_gap[won_col].mean():.4f}")
print()

# Interpretation A: m1_prob is probability for the VALUE fighter
df['m1_pick_won_A'] = np.where(
    df['m1_prob'] > 0.5,
    df[won_col],
    1 - df[won_col]
)
print(f"M1 accuracy (A — m1_prob is for value fighter): {df['m1_pick_won_A'].mean():.4f}")

# Interpretation B: m1_prob is probability for Fighter 1 (red corner, f1_name)
# gap_direction=1 → value fighter = f1; gap_direction=-1 → value fighter = f2
# m1_prob > 0.5 → M1 picks f1
# M1 pick won when:
#   m1 picks f1 AND value=f1 AND value won → won_col=1
#   m1 picks f1 AND value=f2 AND value lost → won_col=0
#   m1 picks f2 AND value=f2 AND value won → won_col=1
#   m1 picks f2 AND value=f1 AND value lost → won_col=0
df['m1_picks_f1'] = df['m1_prob'] > 0.5
df['value_is_f1'] = df['gap_direction'] == 1
df['m1_same_as_value'] = df['m1_picks_f1'] == df['value_is_f1']
df['m1_pick_won_B'] = np.where(df['m1_same_as_value'], df[won_col], 1 - df[won_col])
print(f"M1 accuracy (B — m1_prob is for f1/Red corner):  {df['m1_pick_won_B'].mean():.4f}")
print()

print("=== Correlation checks ===")
print(f"m1_prob  vs {won_col}: {df['m1_prob'].corr(df[won_col]):.4f}")
print(f"m2a_prob vs {won_col}: {df['m2a_prob'].corr(df[won_col]):.4f}")
print()

# Check if m2a_prob > 0.5 matches gap_direction
df['m2a_picks_f1'] = df['m2a_prob'] > 0.5
m2a_agrees_with_gap = (df['m2a_picks_f1'] == df['value_is_f1']).mean()
print(f"Rate m2a_prob > 0.5 agrees with gap_direction: {m2a_agrees_with_gap:.4f}")
print(f"  (if m2a_prob is for value fighter this should be ~1.0)")
print(f"  (if m2a_prob is for f1/Red corner this will be ~0.5)")
print()

# Check m2a_prob vs gap_direction=1: when value=f1, is m2a_prob typically > 0.5?
print(f"When gap_direction=1 (value=f1), m2a_prob > 0.5: {(pos_gap['m2a_prob'] > 0.5).mean():.4f}")
print(f"When gap_direction=-1 (value=f2), m2a_prob > 0.5: {(neg_gap['m2a_prob'] > 0.5).mean():.4f}")
print()

# If m2a_prob is for the value fighter:
#   gap_direction=1 → value is f1 → m2a_prob should be f1's prob
#   gap_direction=-1 → value is f2 → m2a_prob should be f2's prob (so > 0.5 still if M2A picks value)
# If m2a_prob is for f1 (Red corner):
#   gap_direction=1 → value is f1 → m2a_prob > 0.5 makes sense
#   gap_direction=-1 → value is f2 → m2a_prob would typically be < 0.5

# Additional: check the gap column definition
# gap = m2a_prob_for_f1 - f1_novig? OR gap = value_fighter_m2a_prob - value_fighter_novig?
print("=== gap column analysis ===")
print(f"gap mean: {df['gap'].mean():.4f}  (signed)")
print(f"gap_size mean: {df['gap_size'].mean():.4f}  (absolute)")
print(f"gap > 0 when gap_direction= 1: {(pos_gap['gap'] > 0).mean():.4f}")
print(f"gap > 0 when gap_direction=-1: {(neg_gap['gap'] > 0).mean():.4f}")
print(f"gap < 0 when gap_direction=-1: {(neg_gap['gap'] < 0).mean():.4f}")
print()

# m1_m2a_agree meaning: does it mean same pick, or something else?
print("=== Agreement column ===")
print(f"m1_m2a_agree mean: {df['m1_m2a_agree'].mean():.4f}")
print(f"When agree=1, {won_col} mean: {df[df['m1_m2a_agree']==1][won_col].mean():.4f}")
print(f"When agree=0, {won_col} mean: {df[df['m1_m2a_agree']==0][won_col].mean():.4f}")
print()

# Does m1_m2a_agree=1 match (m1_prob > 0.5) == (m2a_prob > 0.5)?
df['both_pick_f1'] = (df['m1_prob'] > 0.5) & (df['m2a_prob'] > 0.5)
df['both_pick_f2'] = (df['m1_prob'] < 0.5) & (df['m2a_prob'] < 0.5)
df['computed_agree'] = (df['both_pick_f1'] | df['both_pick_f2']).astype(int)
match_rate = (df['computed_agree'] == df['m1_m2a_agree']).mean()
print(f"m1_m2a_agree matches (m1>0.5)==(m2a>0.5): {match_rate:.4f}")
print(f"  (1.0 = both probabilities are for the same fighter reference)")
print()

# Sample rows where m1 and m2a disagree on sign (to understand orientation)
print("=== Sample rows where m1_prob and m2a_prob are on opposite sides of 0.5 ===")
cross = df[(df['m1_prob'] > 0.5) != (df['m2a_prob'] > 0.5)].head(5)
show_cols = ['f1_name','f2_name','m1_prob','m2a_prob','gap_direction','gap','closing_odds',won_col,'m1_m2a_agree']
print(cross[[c for c in show_cols if c in cross.columns]].to_string())
print()

# Final verdict
print("=== VERDICT ===")
if match_rate > 0.95:
    print("m1_prob and m2a_prob are BOTH for the same reference fighter (f1/Red corner).")
    print(f"Correct M1 reconstruction: interpretation B (accuracy={df['m1_pick_won_B'].mean():.4f})")
    print(f"The 51% in the ROI script was a RECONSTRUCTION BUG — should be {df['m1_pick_won_B'].mean():.4f}")
else:
    print(f"m1_m2a_agree does NOT match (m1>0.5)==(m2a>0.5) — rate {match_rate:.4f}")
    print("Probabilities may have mixed reference frames. Needs further investigation.")
