#!/usr/bin/env python3
"""
training/fix_fighter_identities.py — 8SI v2 Stage 1 side-discovery, one-time data fix

Found while cross-referencing data/round_stats.parquet coverage against
data/ufc-master.csv: data/career_fights_updated.csv has two distinct,
unrelated fighter-identity problems. Both are fixed here; see
docs/DECISIONS.md for the full investigation and evidence.

1. FAKE_PLACEHOLDER_NAMES (69 names, ~1,104 rows): a single fabricated
   16-fight history (opponents James Saunders, Michael Imperato, Jon
   Williams, Allan Wilson, Hakeem Dawodu, Ousmane Thomas Diagne, ...,
   dated 2011-04-16 through 2025-10-18) stamped verbatim onto 69 different
   "fighter" name rows — including Cris Cyborg, Rampage Jackson, Mirko Cro
   Cop, Minotauro Nogueira, Katlyn Chookagian, and other prominent
   fighters spanning incompatible eras/weight classes/genders, proving
   it's fabricated, not a real shared regional record. compute_career_stats
   ()/compute_qa_stats()/compute_got_finished_rate() group by this exact
   string, so every fight in ufc-master.csv involving one of these 69
   names currently gets career-derived features computed from 100% fake
   data. ufc-master.csv itself has real records for these fighters (e.g.
   Cris Cyborg's actual UFC fights against Amanda Nunes, Felicia Spencer)
   — only career_fights_updated.csv is affected. Fixed by DELETING these
   rows outright; affected fighters fall back to the trainer's existing
   missing-data defaults until their real history is reconstructed (a
   separate, future task — round_stats.parquet has real per-round data for
   their 2015+ fights that a future pass could use).

2. DUPLICATE_IDENTITY_PAIRS (24 pairs): unrelated to the above — real
   fighters whose career is split across two name spellings (nickname,
   typo, capitalization/hyphenation, name-order convention, or a formal
   name change). Two evidence tiers, both required to be free of
   fake-placeholder contamination (neither name in
   FAKE_PLACEHOLDER_NAMES, and shared-opponent counts computed with rows
   naming a fake-placeholder or implausibly-high-degree "opponent"
   excluded — see docs/DECISIONS.md):
     - 10 pairs have BYTE-IDENTICAL fight-history signatures (same
       opponent+date sequence in full) under both spellings — unambiguous
       literal duplication; the rename below collapses them via
       drop_duplicates().
     - 14 pairs have a genuinely different (not identical) set of rows
       under each spelling — a real split career, not literal duplication
       — accepted only at >=8 shared opponents on the cleaned signal.
   Several plausible-looking candidates were individually verified via
   their actual opponent lists and REJECTED as dangerous false positives
   this same process would otherwise have merged: Anderson Silva/
   Wanderlei Silva and Nate Diaz/Nick Diaz (distinct famous fighters
   coincidentally sharing many era-mate opponents — Anderson/Wanderlei
   share 7 opponents, just below this file's >=8 bar), Randy Couture/Ryan
   Couture (real father/son), Jake/Joe Ellenberger (likely brothers),
   Erick Silva/Erik Silva (two different, non-overlapping careers proven
   by opponent lists despite an initially-promising 95.2 fuzzy-name
   score), Alex Pereira/Alice Pereira, Dong Hyun Kim/Dong Hyun Ma, Carlo
   Prater/Carlos Prates, Justin Jaynes/Justin Jones, Martin Buday/Martin
   Day — all common-surname or partial-name coincidences with no real
   corroborating evidence. Fixed by renaming the non-canonical spelling
   to the canonical one (whichever spelling round_stats.parquet's
   ufcstats.com source uses) in BOTH files' fighter-name columns, then
   dropping any rows that become exact duplicates as a result (a no-op
   for the 14 genuine-split-career pairs, since their rows were never
   identical to begin with).

Idempotent — rerunning after these fixes are already applied is a no-op
(the fake names/old spellings no longer exist to match against).
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import DATA

CAREER_PATH = os.path.join(DATA, 'career_fights_updated.csv')
MASTER_PATH = os.path.join(DATA, 'ufc-master.csv')

FAKE_PLACEHOLDER_NAMES = {
    'Alberto Uda', 'Alekander Volkov', 'Alexandra Albu', 'Ali Qaisi', 'An Ying Wang',
    'Antonio Carlos Junior', 'Benny Alloway', 'Bharat Kandare', 'Brianna Van Buren',
    'Bruno Korea', 'CJ Keith', 'CM Punk', 'Caludia Gadelha', 'Caludio Puelles',
    'Cheyanne Buys', 'Cris Cyborg', 'Da Un Jung', 'Da-Un Jung', 'Danny Downes',
    'Dmitrii Smoliakov', 'Dmitry Sosnovskiy', 'Emily Peters Kagan', 'Godofredo Pepey',
    'Grigorii Popov', 'Heather Jo Clark', 'Isabela De Pauda', 'Jimmy Wallhead',
    'Jimy Hettes', 'Joanne Calderwood', 'Jon Olav Einemo', 'Joshua Sampo',
    'Junyong Park', 'KB Bhullar', 'KJ Noons', 'Katlyn Chookagian', 'Kazula Vargas',
    'Krzystof Jotko', 'Lara Procopio', 'Leonardo Augusto Leleco', 'Maia Stevenson',
    'Matthew Riddle', 'Minotauro Nogueira', 'Mirko Cro Cop', 'Montserrat Conejo',
    'Nico Musoke', 'Nina Ansaroff', 'Ode Obsourne', 'Omar Antonio Morales Ferrer',
    'Paddy Holohan', 'Philip De Fries', 'Phillip Hawes', 'Rafael Feijao',
    'Rampage Jackson', 'Raphael Pessoa Nunes', 'Sako Chivitchian', 'Saparbeg Safarov',
    'Seohee Ham', 'Seungwoo Choi', 'TJ Laramie', 'Tecia Torres', 'Tiago Trator',
    'Ulka Sasaki', 'Veronica Macedo', 'Vincente Luque', 'William Patolino',
    'Wuliji Buren', 'Yana Kunitskaya', 'Youssef Zalel', 'Zhalgas Zhamagulov',
}

# (non-canonical spelling, canonical spelling) — canonical chosen to match
# round_stats.parquet's ufcstats.com spelling where exactly one side does.
DUPLICATE_IDENTITY_PAIRS = [
    # 10 byte-identical fight-history duplicates
    ('Rocco Martin', 'Anthony Rocco Martin'),
    ("Don'tale Mayes", "Don'Tale Mayes"),
    ('Luiz Garagorri', 'Eduardo Garagorri'),
    ('Germaine De Randamie', 'Germaine de Randamie'),
    ('Jim Crute', 'Jimmy Crute'),
    ('Kai Kara France', 'Kai Kara-France'),
    ('Polo Reyes', 'Marco Polo Reyes'),
    ('Michelle Waterson', 'Michelle Waterson-Gomez'),
    ('Montserrat Rendon', 'Montse Rendon'),
    ('Waldo Cortes-Acosta', 'Waldo Cortes Acosta'),
    # 14 genuine split-career pairs, >=8 clean shared opponents
    ('Elizeu Dos Santos', 'Elizeu Zaleski dos Santos'),
    ('Marcos Rogerio De Lima', 'Marcos Rogerio de Lima'),
    ('Bobby Green', 'King Green'),
    ('Luci Pudilova', 'Lucie Pudilova'),
    ('Philip Rowe', 'Phil Rowe'),
    ('Danaa Batgerel', 'Batgerel Danaa'),
    ('Joshua Culibao', 'Josh Culibao'),
    (' Jun Yong Park', 'JunYong Park'),
    ('Kalinn Williams', 'Khaos Williams'),
    ('Cong Wang', 'Wang Cong'),
    ('Ian Garry', 'Ian Machado Garry'),
    ('Zachary Reese', 'Zach Reese'),
    ('Na Liang', 'Liang Na'),
    ('Weili Zhang', 'Zhang Weili'),
]


def _remove_fake_rows(career):
    before = len(career)
    out = career[~career['fighter'].isin(FAKE_PLACEHOLDER_NAMES)].copy()
    removed = before - len(out)
    return out, removed


def _apply_renames(df, columns):
    """Rename non-canonical -> canonical spelling in the given columns,
    then drop exact-duplicate rows the rename creates (a genuinely
    duplicated fight record, not a distinct one)."""
    rename_map = dict(DUPLICATE_IDENTITY_PAIRS)
    total_renamed = 0
    for col in columns:
        mask = df[col].isin(rename_map)
        total_renamed += int(mask.sum())
        df.loc[mask, col] = df.loc[mask, col].map(rename_map)
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    return df, total_renamed, dropped


def fix_career_fights(path=CAREER_PATH):
    career = pd.read_csv(path)
    career, n_fake_removed = _remove_fake_rows(career)
    career, n_renamed, n_dupes_dropped = _apply_renames(career, ['fighter', 'opponent'])
    career.to_csv(path, index=False)
    return {'fake_rows_removed': n_fake_removed, 'cells_renamed': n_renamed, 'duplicate_rows_dropped': n_dupes_dropped}


def fix_ufc_master(path=MASTER_PATH):
    master = pd.read_csv(path, low_memory=False)
    master, n_renamed, n_dupes_dropped = _apply_renames(master, ['R_fighter', 'B_fighter'])
    master.to_csv(path, index=False)
    return {'cells_renamed': n_renamed, 'duplicate_rows_dropped': n_dupes_dropped}


def main():
    print('=' * 62)
    print('  8SI v2 — fighter-identity data fix')
    print('=' * 62)
    print('\n[1/2] data/career_fights_updated.csv')
    r1 = fix_career_fights()
    print(f'  Fake placeholder rows removed: {r1["fake_rows_removed"]}')
    print(f'  Cells renamed to canonical spelling: {r1["cells_renamed"]}')
    print(f'  Exact-duplicate rows dropped after rename: {r1["duplicate_rows_dropped"]}')

    print('\n[2/2] data/ufc-master.csv')
    r2 = fix_ufc_master()
    print(f'  Cells renamed to canonical spelling: {r2["cells_renamed"]}')
    print(f'  Exact-duplicate rows dropped after rename: {r2["duplicate_rows_dropped"]}')
    print('=' * 62)


if __name__ == '__main__':
    main()
