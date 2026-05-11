"""Synthetic test-data generator for the Mohometer app.

Produces four CSV files under test_data/:

  test_training.csv    — ~720 rows, has Crust_Thickness target.
                         Use as Model-tab training source. Six geographic
                         regions x ~120 samples each.
  test_validation.csv  — ~240 rows, has Crust_Thickness.
                         Use as Validate-tab held-out dataset.
                         Same six regions, different random samples.
  test_prediction.csv  — ~180 rows, Crust_Thickness BLANKED.
                         Use as Predict-tab unknown-samples dataset.
                         Same shape as training/validation, but no target.
  test_edge_cases.csv  — ~30 hand-crafted rows that exercise:
                         • Forward-only auto-populate (Age_Ma alone)
                         • Reverse-only auto-populate (Era / Period / Stage alone)
                         • Precambrian-only (Eon / Era no age, huge error)
                         • K-Pg boundary (Age_Ma == 66.0 -> older unit)
                         • Conflict (Age_Ma + wrong Period)
                         • Stage alias column (Geologic_Stage instead of Geologic_Age_Label)
                         • Missing chemistry features (Sr without Y, etc.)
                         • Typo in category name ('Ordivician')

Chemistry is synthesised from a latent "true thickness" per row by inverting
the published Profeta-2015 (Sr/Y -> H_km) and Sundell-2021 (La/Yb_N -> H_km)
proxy formulas, then adding noise. So Sr/Y and La/Yb_N really do correlate
with crustal thickness in this synthetic data — proxy + ML models should
both produce reasonable predictions.

Run from the project root:
    python test_data/gen_test_data.py

Seeded with 42 so output is reproducible.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
OUT_DIR = Path(__file__).parent
rng = np.random.default_rng(SEED)

# ── Region archetypes ────────────────────────────────────────────────────────
# Each archetype defines a geographic patch + age window + crustal-thickness
# distribution + tectonic setting + rock-type mix. Samples are drawn from
# uniform / normal distributions inside these bounds.
REGIONS = [
    {
        'name': 'Andes Central',
        'lat': (-30.0, -20.0), 'lon': (-70.0, -65.0),
        'age_ma': (5.0, 25.0),           # Miocene to Pliocene
        'crust_km_mean': 55.0, 'crust_km_sd': 5.0,
        'tectonic': 'Continental arc',
        'arc': 'Andes', 'segment': 'Central Volcanic Zone',
        'domain': 'Cordillera Occidental',
        'dataset': 'Synthetic_Andes',
        'rock_mix': ['intermediate', 'felsic'],
    },
    {
        'name': 'Cascades',
        'lat': (40.0, 48.0), 'lon': (-125.0, -120.0),
        'age_ma': (0.1, 12.0),           # Pliocene to recent
        'crust_km_mean': 35.0, 'crust_km_sd': 3.0,
        'tectonic': 'Continental arc',
        'arc': 'Cascades', 'segment': 'High Cascades',
        'domain': 'Cascade arc',
        'dataset': 'Synthetic_Cascades',
        'rock_mix': ['intermediate', 'mafic'],
    },
    {
        'name': 'Japan',
        'lat': (33.0, 41.0), 'lon': (135.0, 142.0),
        'age_ma': (0.0, 22.0),           # Miocene to recent
        'crust_km_mean': 30.0, 'crust_km_sd': 4.0,
        'tectonic': 'Island arc',
        'arc': 'NE Japan', 'segment': 'Tohoku',
        'domain': 'NE Japan arc',
        'dataset': 'Synthetic_Japan',
        'rock_mix': ['intermediate', 'mafic'],
    },
    {
        'name': 'Aleutians',
        'lat': (50.0, 55.0), 'lon': (-178.0, -160.0),
        'age_ma': (0.0, 5.0),
        'crust_km_mean': 25.0, 'crust_km_sd': 3.0,
        'tectonic': 'Island arc',
        'arc': 'Aleutians', 'segment': 'Central Aleutians',
        'domain': 'Aleutian arc',
        'dataset': 'Synthetic_Aleutians',
        'rock_mix': ['mafic', 'intermediate'],
    },
    {
        'name': 'Tibet (collisional)',
        'lat': (28.0, 32.0), 'lon': (85.0, 95.0),
        'age_ma': (10.0, 55.0),          # Eocene to Miocene
        'crust_km_mean': 65.0, 'crust_km_sd': 5.0,
        'tectonic': 'Continental collision',
        'arc': 'Lhasa Terrane', 'segment': 'Gangdese',
        'domain': 'Tibetan plateau',
        'dataset': 'Synthetic_Tibet',
        'rock_mix': ['felsic', 'intermediate'],
    },
    {
        'name': 'Slave craton (Archean)',
        'lat': (60.0, 65.0), 'lon': (-115.0, -105.0),
        'age_ma': (2500.0, 2900.0),      # Neoarchean - Mesoarchean
        'crust_km_mean': 42.0, 'crust_km_sd': 3.0,
        'tectonic': 'Craton',
        'arc': None, 'segment': None,
        'domain': 'Slave craton',
        'dataset': 'Synthetic_Slave',
        'rock_mix': ['felsic', 'intermediate'],  # TTG suite
    },
]

# Cosmochemical (chondritic) abundances used to compute La/Yb_N from raw La/Yb.
CHON_LA = 0.235
CHON_YB = 0.161
CHON_RATIO = CHON_LA / CHON_YB   # (La/CHON_La)/(Yb/CHON_Yb) = (La/Yb)/CHON_RATIO


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rock_chemistry(rng, rock_class):
    """Pick base major-oxide chemistry for a mafic/intermediate/felsic sample.
    Returns dict with SiO2 / TiO2 / Al2O3 / FeO / MgO / CaO / Na2O / K2O in wt%."""
    if rock_class == 'mafic':
        sio2 = float(rng.normal(50, 2))
        mgo  = float(rng.normal(6.5, 1.8))
        cao  = float(rng.normal(9.5, 1.5))
        feo  = float(rng.normal(9.0, 1.5))
        al2o3 = float(rng.normal(15.5, 1.0))
        tio2 = float(rng.normal(1.4, 0.4))
        na2o = float(rng.normal(2.8, 0.5))
        k2o  = float(rng.normal(0.8, 0.4))
    elif rock_class == 'intermediate':
        sio2 = float(rng.normal(60, 3))
        mgo  = float(rng.normal(3.0, 1.0))
        cao  = float(rng.normal(6.0, 1.2))
        feo  = float(rng.normal(5.5, 1.2))
        al2o3 = float(rng.normal(16.5, 1.0))
        tio2 = float(rng.normal(0.8, 0.3))
        na2o = float(rng.normal(3.5, 0.5))
        k2o  = float(rng.normal(1.8, 0.6))
    else:  # felsic
        sio2 = float(rng.normal(70, 2.5))
        mgo  = float(rng.normal(0.8, 0.5))
        cao  = float(rng.normal(2.5, 0.8))
        feo  = float(rng.normal(2.0, 0.8))
        al2o3 = float(rng.normal(15.0, 1.0))
        tio2 = float(rng.normal(0.35, 0.15))
        na2o = float(rng.normal(3.8, 0.4))
        k2o  = float(rng.normal(3.5, 0.7))
    # Clip non-negative
    return {
        'SiO2': max(sio2, 30.0),
        'TiO2': max(tio2, 0.05),
        'Al2O3': max(al2o3, 8.0),
        'FeO':  max(feo, 0.5),
        'MgO':  max(mgo, 0.1),
        'CaO':  max(cao, 0.5),
        'Na2O': max(na2o, 0.5),
        'K2O':  max(k2o,  0.1),
    }


def _trace_chemistry(rng, h_km_true, rock_class):
    """Synthesise Sr / Y / La / Yb / Ce so that Sr/Y and La/Yb_N hit the
    Profeta-2015 / Sundell-2021 inverse-formula targets for a given true H_km,
    plus realistic noise. Felsic rocks get higher Sr and lower Y (the textbook
    'adakitic' signature) than mafic ones."""
    # Profeta 2015 inverse: Sr/Y target = 0.90 * H_km - 7.25
    sr_y_target = max(0.90 * h_km_true - 7.25, 1.0)
    sr_y = max(float(rng.normal(sr_y_target, sr_y_target * 0.20)), 0.5)
    # Sundell 2021 inverse for La/Yb_N
    layb_n_target = max(np.exp((h_km_true - 6.9) / 17.0), 0.5)
    layb_n = max(float(rng.normal(layb_n_target, layb_n_target * 0.20)), 0.5)
    # Pick Y based on rock class
    if rock_class == 'mafic':
        y = float(rng.uniform(18, 30))
    elif rock_class == 'intermediate':
        y = float(rng.uniform(10, 22))
    else:  # felsic
        y = float(rng.uniform(5, 18))
    sr = sr_y * y
    # Yb in a sensible range
    if rock_class == 'mafic':
        yb = float(rng.uniform(1.4, 2.8))
    elif rock_class == 'intermediate':
        yb = float(rng.uniform(0.9, 2.2))
    else:
        yb = float(rng.uniform(0.4, 1.6))
    # La from La/Yb_raw = La/Yb_N * CHON_RATIO
    la_yb_raw = layb_n * CHON_RATIO
    la = yb * la_yb_raw
    # Ce typically ~ 2 * La
    ce = la * float(rng.uniform(1.8, 2.5))
    return {
        'Sr': sr, 'Y': y, 'La': la, 'Yb': yb, 'Ce': ce,
    }


def _region_sample(rng, region):
    """Draw one sample from a region archetype."""
    lat = float(rng.uniform(*region['lat']))
    lon = float(rng.uniform(*region['lon']))
    age_ma = float(rng.uniform(*region['age_ma']))
    h_km_true = float(rng.normal(region['crust_km_mean'], region['crust_km_sd']))
    h_km_true = max(min(h_km_true, 80.0), 10.0)
    rock_class = rng.choice(region['rock_mix'])
    chem = _rock_chemistry(rng, rock_class)
    chem.update(_trace_chemistry(rng, h_km_true, rock_class))
    chem.update({
        'Lat': lat, 'Lon': lon, 'Age_Ma': age_ma,
        'Crust_Thickness': h_km_true,
        'Tectonic_Setting': region['tectonic'],
        'Arc':     region['arc'],
        'Segment': region['segment'],
        'Geologic_Domain': region['domain'],
        'Dataset': region['dataset'],
        'Rock_Type': rock_class,
    })
    return chem


def synthesise_main_bench(rng, n_per_region):
    """Produce a single DataFrame combining all regions."""
    rows = []
    for region in REGIONS:
        for _ in range(n_per_region):
            row = _region_sample(rng, region)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, 'Sample_ID', [f'SYNTH_{i+1:05d}' for i in range(len(df))])
    cols_order = [
        'Sample_ID', 'Dataset', 'Lat', 'Lon', 'Age_Ma',
        'Tectonic_Setting', 'Arc', 'Segment', 'Geologic_Domain', 'Rock_Type',
        'SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O',
        'Sr', 'Y', 'La', 'Yb', 'Ce',
        'Crust_Thickness',
    ]
    df = df[cols_order]
    return df


def _round_numerics(df, oxide_dp=2, trace_dp=1, age_dp=2):
    out = df.copy()
    for c in ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O']:
        if c in out.columns:
            out[c] = out[c].round(oxide_dp)
    for c in ['Sr', 'Y', 'La', 'Yb', 'Ce']:
        if c in out.columns:
            out[c] = out[c].round(trace_dp)
    for c in ['Lat', 'Lon']:
        if c in out.columns:
            out[c] = out[c].round(4)
    if 'Age_Ma' in out.columns:
        out['Age_Ma'] = out['Age_Ma'].round(age_dp)
    if 'Crust_Thickness' in out.columns:
        out['Crust_Thickness'] = out['Crust_Thickness'].round(1)
    return out


# ── Real-world messy-column bench ────────────────────────────────────────────

# Mapping from canonical internal name to a messy real-world header. Designed
# to exercise the Prepare-tab auto-mapping registry: unit tags in headers,
# parentheses / brackets / spaces, mixed case, abbreviations, ppb-units that
# need conversion, alternative iron expressions, and synonym names.
_MESSY_RENAMES = {
    'Sample_ID':         'Sample',
    'Dataset':           'Reference',
    'Lat':               'Latitude (DD)',
    'Lon':               'Longitude (DD)',
    'Age_Ma':            'Age (Ma)',
    'Tectonic_Setting':  'Tectonic Setting',
    'Arc':               'Arc Name',
    'Segment':           'Volcanic Segment',
    'Geologic_Domain':   'Domain',
    'Rock_Type':         'Lithology',
    'SiO2':              'SiO2 (wt%)',
    'TiO2':              'TiO2 wt%',
    'Al2O3':             'Al2O3 (wt%)',
    'FeO':               'FeOt',           # iron expressed as FeO-total
    'MgO':               'MgO (wt%)',
    'CaO':               'CaO_wt',
    'Na2O':              'Na2O wt%',
    'K2O':               'K2O (%)',
    'Sr':                'Sr [ppm]',
    'Y':                 'Y_ppm',
    'La':                'La (ppm)',
    'Yb':                'Yb_ppm',
    'Ce':                'Ce ppm',
    'Crust_Thickness':   'Moho depth (km)',
}


def real_world_messy_bench(rng):
    """A ~150-row bench with messy real-world column headers, mixed iron
    expressions, ppb-encoded trace elements that need converting to ppm,
    typical missing-value tokens, and some junk columns the registry should
    ignore. Designed to exercise the Prepare-tab auto-mapping pipeline."""
    # Draw 25 samples from each of the 6 regions for a 150-row file.
    rows = []
    for region in REGIONS:
        for _ in range(25):
            rows.append(_region_sample(rng, region))
    df = pd.DataFrame(rows)
    df.insert(0, 'Sample_ID', [f'RW_{i+1:04d}' for i in range(len(df))])
    # Add a few extra columns the registry should map / ignore.
    # 1. Sr_ppb: same Sr but expressed in ppb (× 1000) — tests the ppm/ppb
    #    unit handling. We populate this for half the rows; the other half
    #    keeps Sr in ppm so both code paths fire.
    sr_in_ppb_mask = rng.random(len(df)) < 0.5
    df['Sr_ppb'] = np.where(sr_in_ppb_mask, df['Sr'] * 1000.0, np.nan)
    df.loc[sr_in_ppb_mask, 'Sr'] = np.nan   # only one Sr column populated per row
    # 2. Geologic categories: include one but NOT all — auto-populate
    #    should fill in the missing levels from Age_Ma.
    df['Period'] = pd.NA   # will be ignored (empty) — exercises the empty-column path
    df['Stage']  = pd.NA
    # 3. Junk columns the registry shouldn't touch.
    df['Lab ID']           = [f'L{rng.integers(1000, 9999)}' for _ in range(len(df))]
    df['Date analysed']    = pd.to_datetime('2024-01-01') + pd.to_timedelta(rng.integers(0, 700, len(df)), unit='D')
    df['Comments']         = ''
    # 4. Round the numerics first, THEN sprinkle in real-world missing tokens.
    df = _round_numerics(df)
    # 5. Sprinkle missing-value tokens — about 5% of cells in chemistry
    #    columns get one of {'NA', 'n/a', '-', 'bdl', '<0.1'}. We do this
    #    AFTER rounding so the original numeric distribution is intact.
    missing_tokens = ['NA', 'n/a', '-', 'bdl', '<0.1']
    chem_cols = ['SiO2','TiO2','Al2O3','FeO','MgO','CaO','Na2O','K2O',
                 'Sr','Y','La','Yb','Ce','Sr_ppb']
    for col in chem_cols:
        # Cast to object so we can store string tokens alongside floats
        # without triggering pandas' incompatible-dtype FutureWarning.
        df[col] = df[col].astype(object)
        mask = rng.random(len(df)) < 0.04
        if mask.any():
            df.loc[mask, col] = rng.choice(missing_tokens, size=int(mask.sum()))
    # 6. Now rename to the messy real-world headers. Sr_ppb stays as-is.
    df = df.rename(columns=_MESSY_RENAMES)
    # 7. Reorder so headers appear in a sensible-but-messy order (sample +
    #    location first, then chemistry, then misc).
    preferred_order = [
        'Sample', 'Reference', 'Latitude (DD)', 'Longitude (DD)',
        'Age (Ma)', 'Period', 'Stage',
        'Tectonic Setting', 'Arc Name', 'Volcanic Segment', 'Domain',
        'Lithology',
        'SiO2 (wt%)', 'TiO2 wt%', 'Al2O3 (wt%)', 'FeOt', 'MgO (wt%)',
        'CaO_wt', 'Na2O wt%', 'K2O (%)',
        'Sr [ppm]', 'Sr_ppb', 'Y_ppm', 'La (ppm)', 'Yb_ppm', 'Ce ppm',
        'Moho depth (km)',
        'Lab ID', 'Date analysed', 'Comments',
    ]
    df = df[[c for c in preferred_order if c in df.columns]]
    return df


# ── Edge-case bench (hand-crafted) ──────────────────────────────────────────

def edge_cases():
    """Hand-crafted rows that exercise every quirk of the ICS auto-populate +
    column-mapping pipeline. Documented inline so future readers know which
    behaviour each row exercises."""
    rows = [
        # 1-3: forward-only — Age_Ma alone, categories should auto-fill
        {'Sample_ID': 'EDGE_01_age_only_445',  'Age_Ma':  445.0, 'Note': 'forward: should fill Hirnantian / Ordovician / Paleozoic / Phanerozoic'},
        {'Sample_ID': 'EDGE_02_age_only_120',  'Age_Ma':  120.0, 'Note': 'forward: should fill Aptian / Lower / Cretaceous / Mesozoic'},
        {'Sample_ID': 'EDGE_03_age_only_3500', 'Age_Ma': 3500.0, 'Note': 'forward: should fill Paleoarchean / Archean (no period / epoch / stage)'},
        # 4-5: K-Pg boundary case
        {'Sample_ID': 'EDGE_04_kpg_exact',     'Age_Ma':   66.0, 'Note': 'boundary: should be Maastrichtian (OLDER unit)'},
        {'Sample_ID': 'EDGE_05_kpg_paleocene', 'Age_Ma':   65.5, 'Note': 'boundary: should be Danian / Paleocene'},
        # 6-8: reverse-only — categories filled, no Age_Ma. Stage / Era / Eon levels.
        {'Sample_ID': 'EDGE_06_stage_only',    'Geologic_Age_Label': 'Hirnantian',                  'Note': 'reverse: should derive Age_Ma ~444.5 +/- 0.7'},
        {'Sample_ID': 'EDGE_07_era_precamb',   'Geologic_Era':       'Mesoarchean',                 'Note': 'reverse: should derive Age_Ma ~3000 +/- 200 (huge error)'},
        {'Sample_ID': 'EDGE_08_eon_only',      'Geologic_Eon':       'Proterozoic',                 'Note': 'reverse: should derive Age_Ma from full Proterozoic span'},
        # 9: stage alias column (Geologic_Stage instead of Geologic_Age_Label)
        {'Sample_ID': 'EDGE_09_stage_alias',   'Geologic_Stage':     'Cenomanian',                  'Note': 'reverse via alias column: should coalesce into Geologic_Age_Label'},
        # 10-11: conflict cases (Age_Ma + wrong category)
        {'Sample_ID': 'EDGE_10_conflict_p',    'Age_Ma':  470.0, 'Geologic_Period': 'Devonian',     'Note': 'conflict: 470 Ma is Ordovician, not Devonian'},
        {'Sample_ID': 'EDGE_11_conflict_e',    'Age_Ma':   30.0, 'Geologic_Era':    'Mesozoic',     'Note': 'conflict: 30 Ma is Cenozoic, not Mesozoic'},
        # 12: typo in category name (graceful failure)
        {'Sample_ID': 'EDGE_12_typo',          'Geologic_Period':    'Ordivician',                  'Note': 'typo: should NOT match (graceful NaN)'},
        # 13-14: user value preserved (pre-filled + Age_Ma)
        {'Sample_ID': 'EDGE_13_user_kept',     'Age_Ma':  445.0, 'Geologic_Era': 'Paleozoic',       'Note': 'user value preserved: Paleozoic should NOT be overwritten'},
        {'Sample_ID': 'EDGE_14_user_kept2',    'Age_Ma':  100.0, 'Geologic_Period': 'Cretaceous',   'Note': 'user value preserved: Cretaceous should stay'},
        # 15: case-insensitive lookup
        {'Sample_ID': 'EDGE_15_case_lower',    'Geologic_Period':    'cretaceous',                  'Note': 'case-insensitive: should resolve to Cretaceous'},
        # 16-18: full chemistry + age WITH categories (the common modelling case)
        {'Sample_ID': 'EDGE_16_full',   'Lat': -25.0, 'Lon': -68.0, 'Age_Ma': 12.0,
         'Tectonic_Setting': 'Continental arc', 'Arc': 'Andes', 'Geologic_Era': 'Cenozoic',
         'Rock_Type': 'intermediate',
         'SiO2': 62.5, 'TiO2': 0.7, 'Al2O3': 16.8, 'FeO': 4.9, 'MgO': 2.8,
         'CaO': 5.5, 'Na2O': 3.4, 'K2O': 2.1,
         'Sr': 720, 'Y': 12, 'La': 28, 'Yb': 1.2, 'Ce': 56,
         'Crust_Thickness': 55.0, 'Note': 'full sample: Andes-like thick crust'},
        {'Sample_ID': 'EDGE_17_oceanic','Lat':  52.0, 'Lon': -175.0, 'Age_Ma':  2.0,
         'Tectonic_Setting': 'Island arc', 'Arc': 'Aleutians', 'Geologic_Era': 'Cenozoic',
         'Rock_Type': 'mafic',
         'SiO2': 51.0, 'TiO2': 1.3, 'Al2O3': 15.4, 'FeO': 8.9, 'MgO': 6.5,
         'CaO': 9.6, 'Na2O': 2.9, 'K2O': 0.6,
         'Sr': 320, 'Y': 24, 'La': 8, 'Yb': 2.5, 'Ce': 18,
         'Crust_Thickness': 24.0, 'Note': 'full sample: oceanic-arc thin crust'},
        {'Sample_ID': 'EDGE_18_tibet', 'Lat':  29.5, 'Lon':  90.0, 'Age_Ma': 22.0,
         'Tectonic_Setting': 'Continental collision', 'Geologic_Era': 'Cenozoic',
         'Rock_Type': 'felsic',
         'SiO2': 70.8, 'TiO2': 0.32, 'Al2O3': 14.8, 'FeO': 2.0, 'MgO': 0.7,
         'CaO': 2.3, 'Na2O': 3.9, 'K2O': 3.6,
         'Sr': 850, 'Y': 6, 'La': 42, 'Yb': 0.6, 'Ce': 92,
         'Crust_Thickness': 68.0, 'Note': 'full sample: Tibet-like ultra-thick crust'},
        # 19-21: missing feature subsets — what the readiness diagnostic shows
        {'Sample_ID': 'EDGE_19_no_Sr',  'Age_Ma': 10.0, 'Y': 15, 'La': 20, 'Yb': 1.5, 'Note': 'missing Sr: Profeta Sr/Y proxy NOT computable'},
        {'Sample_ID': 'EDGE_20_no_Yb',  'Age_Ma': 10.0, 'Sr': 600, 'Y': 15, 'La': 25, 'Note': 'missing Yb: Sundell La/Yb_N proxy NOT computable'},
        {'Sample_ID': 'EDGE_21_traces_only', 'Sr': 500, 'Y': 18, 'La': 22, 'Yb': 1.4, 'Ce': 48, 'Note': 'no Lat/Lon, no Age_Ma, no oxides'},
        # 22-23: Cambrian unnamed stages (real ICS names)
        {'Sample_ID': 'EDGE_22_stage10', 'Geologic_Age_Label': 'Stage 10',     'Note': 'Cambrian unnamed Stage 10 - real official ICS name'},
        {'Sample_ID': 'EDGE_23_terren',  'Geologic_Epoch':     'Terreneuvian', 'Note': 'Cambrian Epoch with no formal Stage at this level'},
        # 24-25: Hadean / Pridoli edge cases
        {'Sample_ID': 'EDGE_24_pridoli', 'Geologic_Epoch':     'Pridoli',      'Note': 'Pridoli is Silurian Epoch with no Stage subdivision'},
        {'Sample_ID': 'EDGE_25_hadean',  'Age_Ma':            4200.0,          'Note': 'Hadean: only Eon defined, others should be empty'},
        # 26-28: weird inputs — should fall through without errors
        {'Sample_ID': 'EDGE_26_neg_age', 'Age_Ma':              -5.0,          'Note': 'invalid negative age — no fill, no error'},
        {'Sample_ID': 'EDGE_27_huge',    'Age_Ma':           10000.0,          'Note': 'invalid out-of-range age — no fill, no error'},
        {'Sample_ID': 'EDGE_28_empty',                                          'Note': 'no fields - nothing to fill'},
        # 29-30: regional metadata for grouping tests (mixed Tectonic + Arc + Era)
        {'Sample_ID': 'EDGE_29_kt_japan','Tectonic_Setting': 'Island arc',  'Arc': 'NE Japan',
         'Geologic_Era': 'Cenozoic', 'Age_Ma': 8.0,
         'Note': 'grouping fodder: Japan island arc, Miocene'},
        {'Sample_ID': 'EDGE_30_kt_andes','Tectonic_Setting': 'Continental arc', 'Arc': 'Andes',
         'Geologic_Era': 'Cenozoic', 'Age_Ma': 18.0,
         'Note': 'grouping fodder: Andes continental arc, Miocene'},
    ]
    return pd.DataFrame(rows)


# ── Run ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Generating synthetic test data into {OUT_DIR}/ ...')
    # 720 training / 240 validation / 180 prediction = 6 regions x (120 + 40 + 30)
    train_rng = np.random.default_rng(SEED)
    val_rng   = np.random.default_rng(SEED + 1)
    pred_rng  = np.random.default_rng(SEED + 2)
    training   = _round_numerics(synthesise_main_bench(train_rng, n_per_region=120))
    validation = _round_numerics(synthesise_main_bench(val_rng,   n_per_region=40))
    prediction = _round_numerics(synthesise_main_bench(pred_rng,  n_per_region=30))
    # Blank out the target on the prediction set.
    prediction = prediction.drop(columns=['Crust_Thickness'])

    edges = edge_cases()
    messy_rng = np.random.default_rng(SEED + 3)
    messy = real_world_messy_bench(messy_rng)
    training.to_csv  (OUT_DIR / 'test_training.csv',   index=False)
    validation.to_csv(OUT_DIR / 'test_validation.csv', index=False)
    prediction.to_csv(OUT_DIR / 'test_prediction.csv', index=False)
    edges.to_csv     (OUT_DIR / 'test_edge_cases.csv', index=False)
    messy.to_csv     (OUT_DIR / 'test_real_world_messy.csv', index=False)

    print(f'  test_training.csv          {len(training)} rows  ({training["Dataset"].nunique()} datasets)')
    print(f'  test_validation.csv        {len(validation)} rows')
    print(f'  test_prediction.csv        {len(prediction)} rows  (Crust_Thickness blanked)')
    print(f'  test_edge_cases.csv        {len(edges)} rows')
    print(f'  test_real_world_messy.csv  {len(messy)} rows  ({len(messy.columns)} columns with real-world names)')
    print()
    print('Summary of training set:')
    print(training[['Dataset','Age_Ma','Crust_Thickness','SiO2','MgO','Sr','Y']].describe().round(1))
    print()
    print('Crustal thickness by region (training):')
    print(training.groupby('Dataset')['Crust_Thickness'].agg(['count','mean','std','min','max']).round(1))


if __name__ == '__main__':
    main()
