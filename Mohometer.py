# Mohometer
# Single-file Streamlit app for crustal-thickness estimation from geochemistry.
# Wraps Guo & Yang (2023), Zou et al. (2021), Sundell et al. (2021), Profeta et al. (2015),
# Mantle & Collins (2008), Luffi & Ducea (2022) and CRUST1.0 into one workflow.

from __future__ import annotations
from pathlib import Path
from io import BytesIO
import base64
import html
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.path import Path as MplPath
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Auto-clustering (KMeans/DBSCAN/Agglomerative + GMM two-population detection)
# parked at engine/_parked_auto_clustering.py — see that file for re-enable steps.
try:
    from scipy.spatial import cKDTree, Delaunay, QhullError
    from scipy.interpolate import LinearNDInterpolator as _LinearNDInterp
except Exception:
    cKDTree = None
    Delaunay = None
    QhullError = Exception
    _LinearNDInterp = None
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from engine.grouping import (
    candidate_group_columns as _g_candidate_group_columns,
    candidate_numeric_columns as _g_candidate_numeric_columns,
    numeric_breaks as _g_numeric_breaks,
    assign_numeric_bins as _g_assign_numeric_bins,
    auto_groups_from_filters as _g_auto_groups_from_filters,
    group_stats as _g_group_stats,
    round_trip_columns as _g_round_trip_columns,
    attach_polyline_projections as _g_attach_polyline_projections,
    rolling_window_smoothed as _g_rolling_window_smoothed,
    aggregate_to_groups as _g_aggregate_to_groups,
)

APP_VERSION = 'v10.1 feature-set'
# Resolve default reference-file paths relative to this script's directory
# (NOT the cwd Streamlit happens to be launched from). Otherwise launching
# the app from a different folder makes Path('CRUST_1_0_excel.csv').exists()
# return False even when the file sits next to Mohometer.py — and the
# Validate-tab "Crustal-thickness target" dropdown silently loses CRUST1.0.
_APP_DIR = Path(__file__).resolve().parent
DEFAULT_CRUST1_GRID = _APP_DIR / 'CRUST_1_0_excel.csv'
GAME_CALIBRATION_FILE = _APP_DIR / 'LuffiDucea_2022_Calibration.csv'
DEFAULT_LITHOREF18_XYZ = _APP_DIR / 'LithoRef18.xyz'
DEFAULT_LITHOREF18_ZIP = _APP_DIR / 'Alfonso 2019 supplemental_files.zip'
_ONEDRIVE_LITHOREF18_ZIP = Path(r'C:\Users\david.holder\OneDrive - Barrick Gold Corporation\Documents - Global Exploration\Non Technical data\Training & Reference\ASC\4_2026 Manuals\Crustal Thickness\Papers\Alfonso 2019 supplemental_files.zip')
GAME_ALPHA_DEFAULT = 6.79
GAME_BETA_DEFAULT = 26.40
RESIDUAL_COLORSCALE = [[0.0,'#244575'], [0.48,'#d9e7f0'], [0.50,'#f7f7f4'], [0.52,'#f4dfcf'], [1.0,'#8f1729']]
SUMMARY_MEDIAN_COLOR = '#69c58e'
SUMMARY_MEAN_COLOR = '#f0b36a'
# Shared colorscale and range used for ALL crustal-thickness layers
# (sample scatter, background grid, GAME N-mohometer).
THICKNESS_COLORSCALE = 'Viridis'
THICKNESS_CMIN = 10
THICKNESS_CMAX = 70
GUO_FEATURES = ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Sr','Y','Rb','Ba','Hf','Nb','Ta','Th']
# Complete set of raw element/oxide column names — NO ratios, NO derived or proxy columns.
# Used to restrict "Full suite" and Custom list to measured chemistry only.
ALL_ELEMENT_FEATURES = list(dict.fromkeys([
    # Major oxides
    'SiO2','TiO2','Al2O3','FeO','Fe2O3','Fe2O3T','MnO','MgO','CaO','Na2O','K2O','P2O5','LOI',
    # REE (light → heavy)
    'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
    # HFSE + Y + Th/U
    'Zr','Hf','Nb','Ta','Th','U','Y',
    # LILE
    'Rb','Sr','Ba',
    # Transition metals & other trace
    'Sc','V','Cr','Co','Ni','Cu','Zn','Ga','Pb','Li','Be','Cs',
    # Less common but legitimate measurements
    'Cd','Ag','As','Sb','Tl','In','Mo','W','Sn','Bi',
]))
ZOU_2021_FEATURES = ['SiO2','TiO2','Al2O3','CaO','MgO','MnO','K2O','Na2O','P2O5','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Th','U']
# Elements that underpin every GAME sensor in Luffi & Ducea (2022)
LUFFI_FEATURES = ['SiO2','TiO2','FeO','MnO','MgO','CaO','Na2O','K2O',
                   'Sc','V','Cr','Co','Ni','Ga','Rb','Sr','Y','Zr','Nb',
                   'Ba','La','Ce','Nd','Sm','Gd','Dy','Yb','Lu','Hf','Pb','Th','U']
IMMOBILE_FEATURES = ['TiO2','Al2O3','Y','Zr','Nb','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Th','U']
FEATURE_SETS = {
    'Guo & Yang (2023)': GUO_FEATURES,
    'Zou et al. (2021)': ZOU_2021_FEATURES,
    'Luffi & Ducea (2022)': LUFFI_FEATURES,
    'Immobile elements': IMMOBILE_FEATURES,
}
ALGORITHMS = ['ExtraTrees','RandomForest','GradientBoosting','HistGradientBoosting'] + (['XGBoost'] if XGBRegressor is not None else [])
TRAINING_COLUMNS = ['Lon','Lat','Crust_Thickness','Age_Ma'] + GUO_FEATURES
# Increment this whenever iron_to_feo / std_cols / coerce changes so the
# @st.cache_data on read_table is automatically invalidated on the next run.
_READ_TABLE_VERSION = 2
CHON = {
    # McDonough & Sun (1995) CI chondrite — all REEs + Y + Hf
    'La':0.237,  'Ce':0.612,  'Pr':0.0953, 'Nd':0.457,  'Sm':0.153,
    'Eu':0.058,  'Gd':0.2055, 'Tb':0.0374, 'Dy':0.254,  'Ho':0.0566,
    'Er':0.166,  'Tm':0.0256, 'Yb':0.161,  'Lu':0.0248,
    'Y':1.57,    'Hf':0.104,
}
FE2O3_TO_FEO = 0.8998

ALIASES = {
 'sample':'Sample_ID','sampleid':'Sample_ID','sample_id':'Sample_ID','sample name':'Sample_ID','sample_name':'Sample_ID','sample number':'Sample_ID','sample no':'Sample_ID','id':'Sample_ID',
 'longitude':'Lon','long':'Lon','lon':'Lon','longitude e':'Lon','longitude x':'Lon','latitude':'Lat','lat':'Lat','latitude n':'Lat','latitude y':'Lat',
 'latitude min':'Lat_Min','latitude max':'Lat_Max','longitude min':'Lon_Min','longitude max':'Lon_Max','lat min':'Lat_Min','lat max':'Lat_Max','lon min':'Lon_Min','lon max':'Lon_Max',
 'min age ma':'Age_Min_Ma','max age ma':'Age_Max_Ma',
 'age':'Age_Ma','age ma':'Age_Ma','age_ma':'Age_Ma','absolute age':'Age_Ma','age numeric':'Age_Ma','epoch':'Geologic_Epoch','period':'Geologic_Period','era':'Geologic_Era','stage':'Geologic_Stage','arc':'Arc','segment':'Segment','belt':'Belt','domain':'Geologic_Domain','location':'Location','dataset':'Dataset','citations':'Dataset','geochemistry reference':'Dataset','geochem reference':'Dataset','setting':'Tectonic_Setting','tectonic':'Tectonic_Setting','tectonic setting':'Tectonic_Setting',
 'rock':'Lithology_Type','rock_type':'Lithology_Type','rock type':'Lithology_Type','rock name':'Lithology_Type','lithology':'Lithology_Type','lithology type':'Lithology_Type','crustal thickness':'Crust_Thickness','crust thickness':'Crust_Thickness','moho':'Crust_Thickness','moho depth':'Crust_Thickness','elevation':'Elevation_km','elevation km':'Elevation_km','mean elevation':'Elevation_km','median elevation':'Elevation_km',
 'sio2':'SiO2','tio2':'TiO2','al2o3':'Al2O3','feo':'FeO','feot':'FeO','tfeo':'FeO','totalfeo':'FeO','total feo':'FeO','fe2o3':'Fe2O3','fe2o3t':'Fe2O3T','totalfe2o3':'Fe2O3T','total fe2o3':'Fe2O3T',
 'mno':'MnO','mgo':'MgO','cao':'CaO','ca0':'CaO','na2o':'Na2O','k2o':'K2O','p2o5':'P2O5','loi':'LOI',
 'la':'La','ce':'Ce','pr':'Pr','nd':'Nd','sm':'Sm','eu':'Eu','gd':'Gd','tb':'Tb','dy':'Dy','ho':'Ho','er':'Er','tm':'Tm','yb':'Yb','lu':'Lu','sr':'Sr','y':'Y','rb':'Rb','ba':'Ba','hf':'Hf','nb':'Nb','ta':'Ta','th':'Th','u':'U','zr':'Zr',
 'sc':'Sc','v':'V','cr':'Cr','co':'Co','ni':'Ni','cu':'Cu','zn':'Zn','ga':'Ga','pb':'Pb','li':'Li','be':'Be','cs':'Cs',
 'cd':'Cd','ag':'Ag','as':'As','arsenic':'As','sb':'Sb','antimony':'Sb','tl':'Tl','thallium':'Tl','cadmium':'Cd','silver':'Ag'
}
RATIO_ALIASES = {
 'sr y':'Sr_Y','sry':'Sr_Y','sr/y':'Sr_Y',
 'la yb':'La_Yb_raw','layb':'La_Yb_raw','la/yb':'La_Yb_raw',
 'la yb n':'La_Yb_N','laybn':'La_Yb_N','la/yb n':'La_Yb_N','la ybn':'La_Yb_N',
 'ce y':'Ce_Y','cey':'Ce_Y','ce/y':'Ce_Y',
 'zr y':'Zr_Y','zry':'Zr_Y','zr/y':'Zr_Y',
 'dy yb':'Dy_Yb','dyyb':'Dy_Yb','dy/yb':'Dy_Yb',
 'gd yb':'Gd_Yb','gdyb':'Gd_Yb','gd/yb':'Gd_Yb',
 'sm yb':'Sm_Yb','smyb':'Sm_Yb','sm/yb':'Sm_Yb',
 'ce yb':'Ce_Yb','ceyb':'Ce_Yb','ce/yb':'Ce_Yb',
 'nd yb':'Nd_Yb','ndyb':'Nd_Yb','nd/yb':'Nd_Yb',
 'nb yb':'Nb_Yb','nbyb':'Nb_Yb','nb/yb':'Nb_Yb',
 'th yb':'Th_Yb','thyb':'Th_Yb','th/yb':'Th_Yb',
 'la y':'La_Y','lay':'La_Y','la/y':'La_Y',
 'nd y':'Nd_Y','ndy':'Nd_Y','nd/y':'Nd_Y',
 'nb y':'Nb_Y','nby':'Nb_Y','nb/y':'Nb_Y',
 'th y':'Th_Y','thy':'Th_Y','th/y':'Th_Y',
 'la sm':'La_Sm','lasm':'La_Sm','la/sm':'La_Sm',
 'ba v':'Ba_V','bav':'Ba_V','ba/v':'Ba_V',
 'ba sc':'Ba_Sc','basc':'Ba_Sc','ba/sc':'Ba_Sc',
 'ni sc':'Ni_Sc','nisc':'Ni_Sc','ni/sc':'Ni_Sc',
 'ni v':'Ni_V','niv':'Ni_V','ni/v':'Ni_V',
 'cr sc':'Cr_Sc','crsc':'Cr_Sc','cr/sc':'Cr_Sc',
 'cr v':'Cr_V','crv':'Cr_V','cr/v':'Cr_V',
 'lu hf':'Lu_Hf','luhf':'Lu_Hf','lu/hf':'Lu_Hf',
 'zr ti':'Zr_Ti','zrti':'Zr_Ti','zr/ti':'Zr_Ti',
 'rb sr':'Rb_Sr','rbsr':'Rb_Sr','rb/sr':'Rb_Sr',
}
CRUST_ALIASES = {
 'total crust thickness':'CRUST1_Total_Crust_km','total crustal thickness':'CRUST1_Total_Crust_km','crust thickness':'CRUST1_Total_Crust_km','crustal thickness':'CRUST1_Total_Crust_km','moho depth':'CRUST1_Total_Crust_km','moho':'CRUST1_Total_Crust_km',
 'total crust thickness incl sediments':'CRUST1_Total_Crust_km','total crust thickness including sediments':'CRUST1_Total_Crust_km','moho depth below surface km':'CRUST1_Total_Crust_km',
 'crystalline crust thickness':'CRUST1_Crystalline_Crust_km','crystalline thickness':'CRUST1_Crystalline_Crust_km','crystalline crust':'CRUST1_Crystalline_Crust_km',
 'crystalline crust thicknessweighted vp':'CRUST1_Vp','crystalline crust thicknessweighted density':'CRUST1_Density',
 'vp':'CRUST1_Vp','v p':'CRUST1_Vp','p wave velocity':'CRUST1_Vp','p velocity':'CRUST1_Vp',
 'density':'CRUST1_Density','rho':'CRUST1_Density'
}
GAME_TRACE_FEATURES = ['Sc','V','Cr','Co','Ni','Cu','Zn','Ga','Pb','Li','Be','Cs']
DERIVED_RATIO_COLUMNS = ['Sr_Y','La_Yb_raw','La_Yb_N','Ce_Y','Zr_Y','Dy_Yb','Gd_Yb','Sm_Yb','Ce_Yb','Nd_Yb','Nb_Yb','Th_Yb','La_Y','Nd_Y','Nb_Y','Th_Y','La_Sm','Ba_V','Ba_Sc','Ni_Sc','Ni_V','Cr_Sc','Cr_V','Lu_Hf','Zr_Ti','Rb_Sr']
NUMERIC = list(dict.fromkeys(GUO_FEATURES + GAME_TRACE_FEATURES + DERIVED_RATIO_COLUMNS + ['Fe2O3','Fe2O3T','U','Zr','LOI','Lat','Lon','Lat_Min','Lat_Max','Lon_Min','Lon_Max','Age_Ma','Age_Min_Yrs','Age_Max_Yrs','Age_Min_Ma','Age_Max_Ma','Elevation_km','Crust_Thickness','CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Vp','CRUST1_Density']))

GAME_SENSORS = [
    ('LaYb','La/Yb','La/Yb'), ('CeYb','Ce/Yb','Ce/Yb'), ('NdYb','Nd/Yb','Nd/Yb'),
    ('LaY','La/Y','La/Y'), ('LuHf','Lu/Hf','Lu/Hf'), ('SmYb','Sm/Yb','Sm/Yb'),
    ('NdY','Nd/Y','Nd/Y'), ('NbY','Nb/Y','Nb/Y'), ('CeY','Ce/Y','Ce/Y'),
    ('LaSm','La/Sm','La/Sm'), ('Hf','Hf','Hf (ppm)'), ('ThYb','Th/Yb','Th/Yb'),
    ('MnO','MnO','MnO (wt%)'), ('GdYb','Gd/Yb','Gd/Yb'), ('ZrY','Zr/Y','Zr/Y'),
    ('K2O','K2O','K2O (wt%)'), ('BaV','Ba/V','Ba/V'), ('A','Na2O+K2O','A (wt%)'),
    ('BaSc','Ba/Sc','Ba/Sc'), ('NbYb','Nb/Yb','Nb/Yb'), ('ACaO','(Na2O+K2O)/CaO','A/CaO'),
    ('Sc','Sc','Sc (ppm)'), ('ThY','Th/Y','Th/Y'), ('DyYb','Dy/Yb','Dy/Yb'),
    ('NiSc','Ni/Sc','Ni/Sc'), ('Pb','Pb','Pb (ppm)'), ('Rb','Rb','Rb (ppm)'),
    ('Ba','Ba','Ba (ppm)'), ('CaO','CaO','CaO (wt%)'), ('SrYx','Sr/Y, Rb/Sr 0.05-0.2','Sr/Y'),
    ('CrSc','Cr/Sc','Cr/Sc'), ('Ni','Ni','Ni (ppm)'), ('Sr','Sr','Sr (ppm)'),
    ('FeOt','FeOt','FeOt (wt%)'), ('U','U','U (ppm)'), ('Ga','Ga','Ga (ppm)'),
    ('ZrTi','Zr/Ti','Zr/Ti'), ('NiV','Ni/V','Ni/V'), ('CrV','Cr/V','Cr/V'),
    ('Co','Co','Co (ppm)'), ('Cr','Cr','Cr (ppm)'),
]

# ── Per-sensor metadata from Luffi & Ducea (2022) Table 1 + §5.5.1 ───────────
#
# mgo_lo / mgo_hi : calibrated MgO range (wt%). Outside this range the LOWESS
#                   surface extrapolates and predictions become unreliable.
# diff_class      : differentiation sensitivity per paper §5.5.1
#                   'insensitive'  → safe to use across mixed lithologies
#                   'mild'         → 1 wt% MgO ≈ <5 km Moho difference
#                   'strong'       → 1 wt% MgO can shift estimate >10 km
# r2              : Table 1 R² (h-calibration), used for ranking
# required        : elements that must be present for the sensor to fire.
#                   Set is checked against df.columns; 'OR' tuples mean any
#                   element in the tuple suffices (e.g. FeO or Fe2O3T).
GAME_SENSOR_META = {
    # sensor:   (mgo_lo, mgo_hi, diff_class,    r2,    (required elements))
    'LaYb':    (0,  10, 'mild',         0.973, ('La','Yb')),
    'CeYb':    (0,  10, 'mild',         0.972, ('Ce','Yb')),
    'LuHf':    (0,  10, 'mild',         0.972, ('Lu','Hf')),
    'NdYb':    (0,  10, 'insensitive',  0.971, ('Nd','Yb')),
    'LaY':     (0,  10, 'mild',         0.970, ('La','Y')),
    'NbY':     (0,  10, 'mild',         0.969, ('Nb','Y')),
    'SmYb':    (1,  10, 'insensitive',  0.968, ('Sm','Yb')),
    'CeY':     (1,  10, 'mild',         0.968, ('Ce','Y')),
    'NdY':     (1,  10, 'mild',         0.967, ('Nd','Y')),
    'ThYb':    (0,  10, 'strong',       0.966, ('Th','Yb')),
    'Hf':      (1,  10, 'mild',         0.965, ('Hf',)),
    'LaSm':    (1,  10, 'mild',         0.965, ('La','Sm')),
    'GdYb':    (0,  10, 'insensitive',  0.962, ('Gd','Yb')),
    'BaV':     (0,   8, 'strong',       0.962, ('Ba','V')),
    'MnO':     (1,  10, 'mild',         0.961, ('MnO',)),
    'K2O':     (1,  10, 'mild',         0.961, ('K2O',)),
    'ZrY':     (0,  10, 'mild',         0.960, ('Zr','Y')),
    'BaSc':    (1,  10, 'mild',         0.959, ('Ba','Sc')),
    'A':       (0,  10, 'mild',         0.958, ('Na2O','K2O')),
    'NbYb':    (0,  10, 'mild',         0.958, ('Nb','Yb')),
    'DyYb':    (0,  10, 'insensitive',  0.957, ('Dy','Yb')),
    'ACaO':    (0,  10, 'strong',       0.956, ('Na2O','K2O','CaO')),
    'ThY':     (0,  10, 'strong',       0.956, ('Th','Y')),
    'CaO':     (0,  10, 'mild',         0.952, ('CaO',)),
    'Rb':      (0,  10, 'mild',         0.951, ('Rb',)),
    'NiSc':    (0,  10, 'mild',         0.951, ('Ni','Sc')),
    'Sc':      (0,  10, 'mild',         0.951, ('Sc',)),
    'Ga':      (0,  10, 'mild',         0.950, ('Ga',)),
    'Pb':      (0,  10, 'mild',         0.949, ('Pb',)),
    'Ba':      (0,  10, 'mild',         0.947, ('Ba',)),
    'U':       (0,  10, 'mild',         0.943, ('U',)),
    'FeOt':    (0,  10, 'mild',         0.941, (('FeO','Fe2O3T','Fe2O3'),)),
    'CrSc':    (0,  10, 'mild',         0.941, ('Cr','Sc')),
    'Ni':      (0,  10, 'strong',       0.941, ('Ni',)),
    'SrYx':    (0,  10, 'insensitive',  0.940, ('Sr','Y','Rb')),
    'Sr':      (0,  10, 'insensitive',  0.939, ('Sr',)),
    'ZrTi':    (0,  10, 'mild',         0.932, ('Zr','TiO2')),
    'NiV':     (0,  10, 'mild',         0.928, ('Ni','V')),
    'CrV':     (0,  10, 'mild',         0.928, ('Cr','V')),
    'Co':      (3,   9, 'strong',       0.921, ('Co',)),
    'Cr':      (1,  10, 'strong',       0.918, ('Cr',)),
}

# Top-10 by R² — explicitly endorsed by paper §5.5.2 as a default subset
GAME_TOP10_BY_R2 = ['LaYb','CeYb','LuHf','NdYb','LaY','NbY','SmYb','CeY','NdY','ThYb']

# Subset where MgO has no significant effect — paper §5.5.1
GAME_MGO_INSENSITIVE = [s for s,m in GAME_SENSOR_META.items() if m[2] == 'insensitive']

GAME_PRESET_LABELS = ['Auto from data', 'Top 10 by R²', 'Mafic (MgO 5–10)',
                      'Intermediate (MgO 2–5)', 'Felsic (MgO < 2)',
                      'MgO-insensitive only', 'All 41', 'Custom']

def key(c):
    s = str(c).strip().lower()
    for ch in ['(',')','[',']','°','′','″']:
        s = s.replace(ch,' ')
    s = s.replace('-',' ').replace('_',' ').replace('/',' ')
    s = ' '.join(s.split())
    return s

def compact_key(c):
    return re.sub(r'[^a-z0-9]+','',str(c).strip().lower())

def strip_known_suffixes(token):
    out = token
    suffixes = ['normalized','normalised','icpms','icp','xrf','ppm','ppb','ppt','wtpct','wtpercent','weightpercent','percent','pct','wt']
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if out.endswith(suffix) and len(out) > len(suffix):
                out = out[:-len(suffix)]
                changed = True
    return out

def canonical_header_name(c):
    raw = str(c).strip()
    k = key(raw)
    compact = compact_key(raw)
    slash = raw.lower().replace(' ','')
    cands = [
        k, compact, slash, strip_known_suffixes(compact),
        k.replace(' ppm',''), k.replace(' ppb',''), k.replace(' wt%',''),
        k.replace(' wt pct',''), k.replace(' wt percent',''), k.replace(' pct',''),
        k.replace(' percent',''),
    ]
    for cand in cands:
        if cand in RATIO_ALIASES:
            return RATIO_ALIASES[cand]
    for cand in cands:
        if cand in ALIASES:
            return ALIASES[cand]
    parts = k.split()
    if len(parts) == 2 and parts[1] in ['x','y'] and parts[0] in ALIASES:
        return ALIASES[parts[0]]
    if len(parts) > 1:
        joined = ''.join(parts)
        stripped = strip_known_suffixes(joined)
        if stripped in ALIASES:
            return ALIASES[stripped]
    return None

def source_priority(column_name, target):
    s = compact_key(column_name)
    score = 0
    if target in ['Lat','Lon','Age_Ma'] and any(x in s for x in ['min','max']):
        score += 50
    if 'icpms' in s:
        score -= 20
    elif 'icp' in s:
        score -= 15
    if 'xrf' in s:
        score -= 8
    if 'feot' in s or 'total' in s:
        score -= 4
    if s == compact_key(target):
        score -= 30
    return score

def std_cols(df):
    out = df.copy()
    mapped = {}
    mapped_sources = set()
    for c in out.columns:
        t = canonical_header_name(c)
        if t:
            mapped.setdefault(t, []).append(c)
            mapped_sources.add(c)
    if not mapped:
        return ensure_unique_columns(out)
    result_cols = {}
    for c in out.columns:
        if c not in mapped_sources:
            result_cols[str(c)] = out[c]
    for target, sources in mapped.items():
        ordered = sorted(sources, key=lambda s: source_priority(s, target))
        series = out[ordered[0]].copy()
        for source in ordered[1:]:
            series = series.combine_first(out[source])
        result_cols[target] = series
    result = pd.DataFrame(result_cols, index=out.index).copy()
    return ensure_unique_columns(result)

def std_crust_cols(df):
    out = std_cols(df)
    ren = {}
    used = set(out.columns)
    for c in out.columns:
        k = key(c)
        cands = [k,k.replace(' ',''),k.replace(' km',''),k.replace(' km s',''),k.replace(' g cm3',''),k.replace(' kg m3','')]
        for x in cands:
            if x in CRUST_ALIASES:
                t = CRUST_ALIASES[x]
                if t not in used or c == t:
                    ren[c] = t
                    used.add(t)
                break
    return ensure_unique_columns(out.rename(columns=ren))

def ensure_unique_columns(df):
    out = df.copy()
    seen = {}
    cols = []
    for c in out.columns:
        base = str(c)
        n = seen.get(base, 0)
        cols.append(base if n == 0 else f'{base}_{n+1}')
        seen[base] = n + 1
    out.columns = cols
    return out

def plot_df(df):
    return ensure_unique_columns(df.reset_index(drop=True))

def col_series(df, col):
    if col not in df:
        return pd.Series(np.nan, index=df.index)
    vals = df.loc[:, col]
    if isinstance(vals, pd.DataFrame):
        vals = vals.iloc[:, 0]
    return vals

def numeric_series(df, col):
    return pd.to_numeric(col_series(df, col), errors='coerce')

def has_numeric_column(df, col):
    return col in df and numeric_series(df, col).notna().any()

def typed_slider(ui, label, min_value, max_value, value, step, key, disabled=False, help=None):
    slider_value = ui.slider(label, min_value, max_value, value, step, key=f'{key}_slider', disabled=disabled, help=help)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(slider_value)
    return float(slider_value)

def typed_range_slider(ui, label, min_value, max_value, value, key, step=None, disabled=False, help=None):
    if step is None:
        step = 1 if all(isinstance(v, int) for v in [min_value, max_value, value[0], value[1]]) else 1.0
    slider_value = ui.slider(label, min_value, max_value, value, step, key=f'{key}_slider', disabled=disabled, help=help)
    c1, c2 = ui.columns(2)
    low = c1.number_input(
        'Min',
        min_value=min_value,
        max_value=max_value,
        value=slider_value[0],
        step=step,
        key=f'{key}_min_typed',
        disabled=disabled,
    )
    high = c2.number_input(
        'Max',
        min_value=min_value,
        max_value=max_value,
        value=slider_value[1],
        step=step,
        key=f'{key}_max_typed',
        disabled=disabled,
    )
    if low > high:
        low, high = high, low
    return (low, high)

GEO_TIME_BINS = [
    ('Holocene','Quaternary','Cenozoic',0.0,0.0117),
    ('Pleistocene','Quaternary','Cenozoic',0.0117,2.58),
    ('Pliocene','Neogene','Cenozoic',2.58,5.333),
    ('Miocene','Neogene','Cenozoic',5.333,23.03),
    ('Oligocene','Paleogene','Cenozoic',23.03,33.9),
    ('Eocene','Paleogene','Cenozoic',33.9,56.0),
    ('Paleocene','Paleogene','Cenozoic',56.0,66.0),
    ('Late Cretaceous','Cretaceous','Mesozoic',66.0,100.5),
    ('Early Cretaceous','Cretaceous','Mesozoic',100.5,145.0),
    ('Late Jurassic','Jurassic','Mesozoic',145.0,163.5),
    ('Middle Jurassic','Jurassic','Mesozoic',163.5,174.7),
    ('Early Jurassic','Jurassic','Mesozoic',174.7,201.4),
    ('Late Triassic','Triassic','Mesozoic',201.4,237.0),
    ('Middle Triassic','Triassic','Mesozoic',237.0,247.2),
    ('Early Triassic','Triassic','Mesozoic',247.2,251.9),
    ('Lopingian','Permian','Paleozoic',251.9,259.5),
    ('Guadalupian','Permian','Paleozoic',259.5,273.0),
    ('Cisuralian','Permian','Paleozoic',273.0,298.9),
    ('Pennsylvanian','Carboniferous','Paleozoic',298.9,323.2),
    ('Mississippian','Carboniferous','Paleozoic',323.2,358.9),
    ('Late Devonian','Devonian','Paleozoic',358.9,382.7),
    ('Middle Devonian','Devonian','Paleozoic',382.7,393.3),
    ('Early Devonian','Devonian','Paleozoic',393.3,419.2),
    ('Silurian','Silurian','Paleozoic',419.2,443.8),
    ('Ordovician','Ordovician','Paleozoic',443.8,485.4),
    ('Cambrian','Cambrian','Paleozoic',485.4,538.8),
    ('Neoproterozoic','Neoproterozoic','Proterozoic',538.8,1000.0),
    ('Mesoproterozoic','Mesoproterozoic','Proterozoic',1000.0,1600.0),
    ('Paleoproterozoic','Paleoproterozoic','Proterozoic',1600.0,2500.0),
    ('Archean','Archean','Archean',2500.0,4000.0),
    ('Hadean','Hadean','Hadean',4000.0,4567.0),
]
GEO_TIME_ALIASES = {
    'recent':'Holocene','modern':'Holocene','quaternary':'Pleistocene','pleistocene':'Pleistocene','holocene':'Holocene',
    'pliocene':'Pliocene','miocene':'Miocene','oligocene':'Oligocene','eocene':'Eocene','paleocene':'Paleocene',
    'neogene':'Miocene','palaeogene':'Eocene','paleogene':'Eocene','cenozoic':'Miocene',
    'cretaceous':'Late Cretaceous','late cretaceous':'Late Cretaceous','early cretaceous':'Early Cretaceous',
    'jurassic':'Middle Jurassic','late jurassic':'Late Jurassic','middle jurassic':'Middle Jurassic','early jurassic':'Early Jurassic',
    'triassic':'Middle Triassic','late triassic':'Late Triassic','middle triassic':'Middle Triassic','early triassic':'Early Triassic',
    'permian':'Guadalupian','carboniferous':'Pennsylvanian','devonian':'Middle Devonian','silurian':'Silurian','ordovician':'Ordovician','cambrian':'Cambrian',
    'paleozoic':'Devonian','palaeozoic':'Devonian','mesozoic':'Jurassic','proterozoic':'Mesoproterozoic','archean':'Archean','archaean':'Archean','hadean':'Hadean',
}
ROCK_TEXT_CLASS = {
    'basalt':'mafic','basanite':'mafic','tephrite':'mafic','gabbro':'mafic','dolerite':'mafic','diabase':'mafic','picrite':'mafic','komatiite':'mafic','mafic':'mafic',
    'andesite':'intermediate','basaltic andesite':'intermediate','trachyandesite':'intermediate','latite':'intermediate','diorite':'intermediate','monzonite':'intermediate','intermediate':'intermediate',
    'dacite':'felsic','rhyolite':'felsic','rhyodacite':'felsic','trachyte':'felsic','granite':'felsic','granodiorite':'felsic','tonalite':'felsic','felsic':'felsic',
    'peridotite':'ultramafic','dunite':'ultramafic','harzburgite':'ultramafic','lherzolite':'ultramafic','ultramafic':'ultramafic',
}

def classify_rock_text(value):
    if pd.isna(value):
        return np.nan
    text = key(value)
    for term, cls in sorted(ROCK_TEXT_CLASS.items(), key=lambda kv: -len(kv[0])):
        if term in text:
            return cls
    return np.nan

def geologic_from_age(age):
    if pd.isna(age):
        return pd.Series({'Geologic_Epoch':'unknown','Geologic_Period':'unknown','Geologic_Era':'unknown'})
    a = float(age)
    for epoch, period, era, young, old in GEO_TIME_BINS:
        if a >= young and a < old:
            return pd.Series({'Geologic_Epoch':epoch,'Geologic_Period':period,'Geologic_Era':era})
    return pd.Series({'Geologic_Epoch':'unknown','Geologic_Period':'unknown','Geologic_Era':'unknown'})

def geologic_midpoint(label):
    if pd.isna(label):
        return np.nan
    norm = key(label)
    epoch_name = GEO_TIME_ALIASES.get(norm)
    if epoch_name is None:
        for epoch, period, era, young, old in GEO_TIME_BINS:
            if norm in [key(epoch),key(period),key(era)]:
                epoch_name = epoch
                break
    if epoch_name is None:
        return np.nan
    for epoch, period, era, young, old in GEO_TIME_BINS:
        if epoch == epoch_name:
            return float((young + old) / 2)
    return np.nan

def add_geologic_time_categories(df):
    out = df.copy()
    source_text = None
    for c in ['Geologic_Age','Geologic_Epoch','Geologic_Period','Geologic_Era','Geologic_Stage']:
        if c in out:
            source_text = c
            break
    if source_text and 'Age_Ma' not in out:
        inferred = out[source_text].map(geologic_midpoint)
        if inferred.notna().any():
            out['Age_Ma'] = inferred
            out['Age_Ma_Source'] = f'inferred from {source_text}'
    elif source_text and 'Age_Ma' in out:
        inferred = out[source_text].map(geologic_midpoint)
        numeric_age = pd.to_numeric(out['Age_Ma'], errors='coerce')
        fill_mask = numeric_age.isna() & inferred.notna()
        if fill_mask.any():
            out.loc[fill_mask,'Age_Ma'] = inferred[fill_mask]
            out.loc[fill_mask,'Age_Ma_Source'] = f'inferred from {source_text}'
        if 'Age_Ma_Source' not in out:
            out['Age_Ma_Source'] = np.where(pd.to_numeric(out['Age_Ma'],errors='coerce').notna(),'numeric','')
    if 'Age_Ma' in out:
        cats = pd.DataFrame([geologic_from_age(v).to_dict() for v in out['Age_Ma']], index=out.index)
        for c in ['Geologic_Epoch','Geologic_Period','Geologic_Era']:
            if c not in out:
                out[c] = cats[c]
            else:
                out[c] = out[c].fillna(cats[c])
    if source_text:
        if 'Geologic_Age_Label' not in out:
            out['Geologic_Age_Label'] = out[source_text].astype(str)
    return out

@st.cache_data(show_spinner=False,max_entries=32)
def tidy_numbers(df):
    out = ensure_unique_columns(df.copy())
    one_dp_names = ['Age_Ma','Crust_Thickness','Observed_km','Predicted_km','Residual_km','RMSE_km','MAE_km','Bias_km','Delta_km','CRUST1_Benchmark_km','CRUST1_Match_Distance_km','Training_2SD_km','Blind_2SD_km','Combined_2SD_km']
    one_dp_names += [c for c in out.columns if str(c).endswith('_km') or str(c).startswith('Modelled_')]
    for c in dict.fromkeys(one_dp_names):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors='coerce').round(1)
    if 'R2' in out:
        out['R2'] = pd.to_numeric(out['R2'], errors='coerce').round(3)
    for c in [c for c in out.columns if c.endswith('_percent') or c.endswith('_pct')]:
        out[c] = pd.to_numeric(out[c], errors='coerce').round(1)
    return out

def table_action_card(title, df, filename=None, key=None):
    if df is None or df.empty:
        st.caption(f'{title}: no rows')
        return
    view = tidy_numbers(plot_df(df))
    safe_title = html.escape(str(title))
    safe_key = compact_key(key or title or 'table')
    filename = filename or f'{safe_key}.csv'
    rows, cols = view.shape
    table_html = view.to_html(index=False, border=0, classes='data-table')
    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; color: #1f2937; }}
h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
.meta {{ color: #6b7280; font-size: 12px; margin-bottom: 14px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th {{ position: sticky; top: 0; background: #f3f4f6; border-bottom: 1px solid #d1d5db; text-align: left; }}
td, th {{ padding: 5px 7px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; }}
tbody tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<div class="meta">{rows:,} rows x {cols:,} columns</div>
{table_html}
</body>
</html>"""
    payload = base64.b64encode(doc.encode('utf-8')).decode('ascii')
    c1,c2,c3 = st.columns([1.4,0.42,0.32])
    c1.caption(f'{title}: {rows:,} rows x {cols:,} columns')
    c2.markdown(
        f'<a href="data:text/html;base64,{payload}" target="_blank" rel="noopener" '
        'style="display:inline-block;padding:0.45rem 0.7rem;border:1px solid #d1d5db;'
        'border-radius:0.45rem;text-decoration:none;color:#1f2937;background:#fff;'
        'font-size:0.88rem;line-height:1.1;">Open table</a>',
        unsafe_allow_html=True,
    )
    c3.download_button('CSV', view.to_csv(index=False).encode('utf-8'), filename, 'text/csv', key=f'table_csv_{safe_key}')

def qualitative_palette(name):
    palettes = {
        'Plotly': px.colors.qualitative.Plotly,
        'Set2': px.colors.qualitative.Set2,
        'Dark24': px.colors.qualitative.Dark24,
        'Alphabet': px.colors.qualitative.Alphabet,
        'Safe': px.colors.qualitative.Safe,
    }
    return palettes.get(name, px.colors.qualitative.Plotly)

def hover_cols(df, candidates, *excluded):
    excluded = {c for c in excluded if c}
    cols = []
    for c in candidates:
        if c in df and c not in excluded and c not in cols:
            cols.append(c)
    return cols

def dms_to_float(v):
    if pd.isna(v): return np.nan
    if isinstance(v,(int,float,np.number)): return float(v)
    s = str(v).strip().replace('°',' ').replace('′',' ').replace("'",' ').replace('″',' ').replace('"',' ')
    neg = any(x in s.upper() for x in ['S','W'])
    vals = []
    for p in s.replace(',',' ').split():
        try: vals.append(float(p))
        except Exception: pass
    if not vals: return np.nan
    x = vals[0] + (vals[1]/60 if len(vals)>1 else 0) + (vals[2]/3600 if len(vals)>2 else 0)
    return -x if neg else x

def coerce(df):
    out = df.copy()
    for c in ['Lat_Min','Lat_Max','Lon_Min','Lon_Max','Age_Min_Yrs','Age_Max_Yrs','Age_Min_Ma','Age_Max_Ma']:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    if 'Lat' not in out and {'Lat_Min','Lat_Max'}.issubset(out):
        out['Lat'] = out[['Lat_Min','Lat_Max']].mean(axis=1)
    if 'Lon' not in out and {'Lon_Min','Lon_Max'}.issubset(out):
        out['Lon'] = out[['Lon_Min','Lon_Max']].mean(axis=1)
    if 'Age_Ma' not in out:
        if {'Age_Min_Ma','Age_Max_Ma'}.issubset(out):
            out['Age_Ma'] = out[['Age_Min_Ma','Age_Max_Ma']].mean(axis=1)
        elif {'Age_Min_Yrs','Age_Max_Yrs'}.issubset(out):
            out['Age_Ma'] = out[['Age_Min_Yrs','Age_Max_Yrs']].mean(axis=1) / 1_000_000.0
    if 'Lat' in out: out['Lat'] = out['Lat'].apply(dms_to_float)
    if 'Lon' in out: out['Lon'] = out['Lon'].apply(dms_to_float)
    for c in NUMERIC:
        if c in out and c not in ['Lat','Lon']:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return add_geologic_time_categories(out)

def iron_to_feo(df):
    out = df.copy(); idx = out.index
    for c in ['FeO','Fe2O3','Fe2O3T']:
        if c in out: out[c] = pd.to_numeric(out[c], errors='coerce')
    feo = out['FeO'].copy() if 'FeO' in out else pd.Series(np.nan,index=idx,dtype=float)
    src = pd.Series('No iron column', index=idx, dtype=object)
    if 'FeO' in out: src.loc[feo.notna()] = 'FeO/FEOT/TFeO used directly as FeO-equivalent'
    if 'Fe2O3' in out:
        f2 = out['Fe2O3']; comb = feo.fillna(0) + FE2O3_TO_FEO*f2.fillna(0); comb.loc[feo.isna() & f2.isna()] = np.nan
        m = f2.notna(); feo.loc[m] = comb.loc[m]
        src.loc[m] = 'FeO + 0.8998*Fe2O3' if 'FeO' in out else '0.8998*Fe2O3 converted to FeO-equivalent'
    if 'Fe2O3T' in out:
        f2t = out['Fe2O3T']; conv = FE2O3_TO_FEO*f2t
        m = (feo.isna() if 'FeO' in out else pd.Series(True,index=idx)) & f2t.notna()
        feo.loc[m] = conv.loc[m]; src.loc[m] = '0.8998*Fe2O3T converted to FeO-equivalent'
    if 'FeO' in out or 'Fe2O3' in out or 'Fe2O3T' in out:
        out['FeO'] = feo; out['FeO_Source'] = src
    return out

def looks_header(vals):
    vals = [key(v) for v in vals if pd.notna(v)]
    joined = ' '.join(vals)
    normalized = set()
    for v in vals:
        canonical = canonical_header_name(v)
        if canonical:
            normalized.add(canonical)
    hits = sum(1 for f in ['SiO2','TiO2','Al2O3','MgO','La','Ce','Nd','Sr','Y','Th'] if f in normalized)
    location_hits = any(v in vals for v in ['longitude x','longitude','lon','latitude y','latitude','lat'])
    sample_hits = any(v in vals for v in ['sample','sample id','sample name','age ma','geologycal age','geological age'])
    target_hits = 'crustal thickness' in joined or 'crust thickness' in joined or 'moho' in joined
    return hits >= 6 and (target_hits or location_hits or sample_hits)

@st.cache_data(show_spinner=False)
def read_table(file_or_path, guo_no_header=False, expected=None, read_ver=_READ_TABLE_VERSION):
    name = getattr(file_or_path,'name',str(file_or_path)).lower()
    if name.endswith(('.xlsx','.xls')):
        raw = pd.read_excel(file_or_path, header=None)
    else:
        raw = pd.read_csv(file_or_path, header=None if guo_no_header else 0)
        if not guo_no_header: raw = std_cols(raw); raw = coerce(raw); raw = iron_to_feo(raw); return raw
    hdr = None
    for i in range(min(len(raw),30)):
        if looks_header(raw.iloc[i].tolist()): hdr = i; break
    if hdr is not None:
        headers = [str(v).strip() if pd.notna(v) else f'Unnamed_{j}' for j,v in enumerate(raw.iloc[hdr].tolist())]
        df = raw.iloc[hdr+1:].copy(); df.columns = headers; df = df.dropna(how='all').reset_index(drop=True)
    elif guo_no_header and expected:
        df = raw.copy(); n = df.shape[1]
        if n == len(expected): df.columns = expected
        elif n >= len(GUO_FEATURES): df.columns = GUO_FEATURES + [f'Extra_{i}' for i in range(n-len(GUO_FEATURES))]
    else:
        df = pd.read_excel(file_or_path) if name.endswith(('.xlsx','.xls')) else raw
    df = std_cols(df)
    if 'Sample_ID' not in df: df.insert(0,'Sample_ID',[f'Sample_{i+1}' for i in range(len(df))])
    df = coerce(df); df = iron_to_feo(df)
    return df

# ─── Data Preparation helpers ─────────────────────────────────────────────────
# Registry: (display_label, internal_name, category, default_unit, conv)
#
#   display_label  — what the user picks in the editor's "Map to" dropdown.
#                    Bare element/oxide names so they match every formula in
#                    this file (no '[wt%]' / '[ppm]' suffix).
#   internal_name  — what the column gets renamed to in the processed
#                    dataframe. This is the string every formula references
#                    (`out['SiO2']`, `out['FeO']`, `GUO_FEATURES`, ...).
#   category       — registry-side grouping for type derivation.
#   default_unit   — the unit auto-applied if the original header has no
#                    detectable unit hint. Also the canonical unit that
#                    values are converted INTO before rename.
#   conv           — None | ('elem_oxide', factor) | ('utm', None)
#                    elem_oxide factor is the molar mass ratio applied when
#                    the user supplies elemental concentration but the model
#                    expects the oxide form (e.g. Al → Al2O3 × 1.8895).
#
# Unit conversion at apply time (combines unit + elem_oxide factors):
#     value × UNIT_TO_PPM[source_unit] / UNIT_TO_PPM[target_unit] × oxide_factor
# where UNIT_TO_PPM = {'ppb': 1e-3, 'ppm': 1.0, 'wt%': 1e4}.
# So `Al ppm` mapped to display `Al` (registry: Al → Al2O3, factor 1.8895,
# default 'wt%') becomes Al2O3 wt% via × (1.0/1e4) × 1.8895 = × 1.8895e-4.
_DP_REGISTRY = [
    # (display_label, internal_name, category, default_unit, conv)
    # Metadata
    ('Sample ID',         'Sample_ID',          'meta', None, None),
    ('Dataset',           'Dataset',            'meta', None, None),
    ('Reference',         'Reference',          'meta', None, None),
    ('Notes / Comments',  'Notes',              'meta', None, None),
    ('Analytical Total',  'Analytical_Total',   'meta', 'wt%', None),
    # Location
    ('Lat',               'Lat',                'location', 'DD', None),
    ('Lon',               'Lon',                'location', 'DD', None),
    ('UTM Easting',       'UTM_Easting',        'location', 'm',  ('utm', None)),
    ('UTM Northing',      'UTM_Northing',       'location', 'm',  ('utm', None)),
    ('Elevation',         'Elevation_km',       'location', 'km', None),
    ('Lat min',           'Lat_Min',            'location', 'DD', None),
    ('Lat max',           'Lat_Max',            'location', 'DD', None),
    ('Lon min',           'Lon_Min',            'location', 'DD', None),
    ('Lon max',           'Lon_Max',            'location', 'DD', None),
    # Age
    ('Age',               'Age_Ma',             'age', 'Ma', None),
    ('Age error',         'Age_Err_Ma',         'age', 'Ma', None),
    ('Age min',           'Age_Min_Ma',         'age', 'Ma', None),
    ('Age max',           'Age_Max_Ma',         'age', 'Ma', None),
    ('Era',               'Geologic_Era',       'age', None, None),
    ('Period',            'Geologic_Period',    'age', None, None),
    ('Epoch',             'Geologic_Epoch',     'age', None, None),
    ('Stage',             'Geologic_Stage',     'age', None, None),
    # Classification / Grouping
    ('Tectonic Setting',  'Tectonic_Setting',   'class', None, None),
    ('Arc',               'Arc',                'class', None, None),
    ('Segment',           'Segment',            'class', None, None),
    ('Domain',            'Geologic_Domain',    'class', None, None),
    ('Belt',              'Belt',               'class', None, None),
    ('Sub-belt',          'Sub_Belt',           'class', None, None),
    ('Location',          'Location',           'class', None, None),
    ('Lithology',         'Lithology_Type',     'class', None, None),
    ('Lithology Grouping', 'Lithology_Grouping','class', None, None),
    ('Country',           'Country',            'class', None, None),
    # Target
    ('Crustal thickness', 'Crust_Thickness',    'target', 'km', None),
    # Major oxides and the elemental forms that get converted to oxides
    ('SiO2',  'SiO2',  'major', 'wt%', None),
    ('Si',    'SiO2',  'major', 'wt%', ('elem_oxide', 2.1394)),
    ('TiO2',  'TiO2',  'major', 'wt%', None),
    ('Ti',    'TiO2',  'major', 'wt%', ('elem_oxide', 1.6685)),
    ('Al2O3', 'Al2O3', 'major', 'wt%', None),
    ('Al',    'Al2O3', 'major', 'wt%', ('elem_oxide', 1.8895)),
    ('FeOt',  'FeO',   'major', 'wt%', None),
    ('FeO',   'FeO',   'major', 'wt%', None),
    ('Fe2O3t','Fe2O3T','major', 'wt%', None),
    ('Fe2O3', 'Fe2O3', 'major', 'wt%', None),
    ('Fe',    'FeO',   'major', 'wt%', ('elem_oxide', 1.2865)),
    ('MnO',   'MnO',   'major', 'wt%', None),
    ('Mn',    'MnO',   'major', 'wt%', ('elem_oxide', 1.2912)),
    ('MgO',   'MgO',   'major', 'wt%', None),
    ('Mg',    'MgO',   'major', 'wt%', ('elem_oxide', 1.6589)),
    ('CaO',   'CaO',   'major', 'wt%', None),
    ('Ca',    'CaO',   'major', 'wt%', ('elem_oxide', 1.3992)),
    ('Na2O',  'Na2O',  'major', 'wt%', None),
    ('Na',    'Na2O',  'major', 'wt%', ('elem_oxide', 1.3479)),
    ('K2O',   'K2O',   'major', 'wt%', None),
    ('K',     'K2O',   'major', 'wt%', ('elem_oxide', 1.2047)),
    ('P2O5',  'P2O5',  'major', 'wt%', None),
    ('P',     'P2O5',  'major', 'wt%', ('elem_oxide', 2.2915)),
    ('LOI',   'LOI',   'major', 'wt%', None),
    ('H2Ot',  'H2Ot',  'major', 'wt%', None),
    # REE — bare names. ppm/ppb is selected via the Unit column at apply time;
    # ppb values are auto-scaled to ppm by the unit-conversion logic.
    ('La','La','ree','ppm',None),
    ('Ce','Ce','ree','ppm',None),
    ('Pr','Pr','ree','ppm',None),
    ('Nd','Nd','ree','ppm',None),
    ('Sm','Sm','ree','ppm',None),
    ('Eu','Eu','ree','ppm',None),
    ('Gd','Gd','ree','ppm',None),
    ('Tb','Tb','ree','ppm',None),
    ('Dy','Dy','ree','ppm',None),
    ('Ho','Ho','ree','ppm',None),
    ('Er','Er','ree','ppm',None),
    ('Tm','Tm','ree','ppm',None),
    ('Yb','Yb','ree','ppm',None),
    ('Lu','Lu','ree','ppm',None),
    # HFSE / LIL / compatible trace — same pattern as REE.
    ('Rb','Rb','trace','ppm',None),
    ('Sr','Sr','trace','ppm',None),
    ('Y', 'Y', 'trace','ppm',None),
    ('Zr','Zr','trace','ppm',None),
    ('Nb','Nb','trace','ppm',None),
    ('Ba','Ba','trace','ppm',None),
    ('Hf','Hf','trace','ppm',None),
    ('Ta','Ta','trace','ppm',None),
    ('Pb','Pb','trace','ppm',None),
    ('Th','Th','trace','ppm',None),
    ('U', 'U', 'trace','ppm',None),
    ('Sc','Sc','trace','ppm',None),
    ('V', 'V', 'trace','ppm',None),
    ('Cr','Cr','trace','ppm',None),
    ('Co','Co','trace','ppm',None),
    ('Ni','Ni','trace','ppm',None),
    ('Cu','Cu','trace','ppm',None),
    ('Zn','Zn','trace','ppm',None),
    ('Ga','Ga','trace','ppm',None),
    ('Li','Li','trace','ppm',None),
    ('Be','Be','trace','ppm',None),
    ('Cs','Cs','trace','ppm',None),
    ('Bi','Bi','trace','ppm',None),
    ('Sn','Sn','trace','ppm',None),
    ('W', 'W', 'trace','ppm',None),
    ('Mo','Mo','trace','ppm',None),
    ('Ge','Ge','trace','ppm',None),
    ('Cd','Cd','trace','ppm',None),
    ('Ag','Ag','trace','ppm',None),
    ('As','As','trace','ppm',None),
    ('Sb','Sb','trace','ppm',None),
    ('Tl','Tl','trace','ppm',None),
]

# Canonical concentration-unit scale: factor to convert any listed unit → ppm
# (the "intermediate" canonical concentration). Other units are passthrough.
_DP_UNIT_TO_PPM = {
    'ppb': 1e-3,
    'ppm': 1.0,
    'wt%': 1e4,
}
# Concentration units the editor's Unit dropdown will offer for numeric chem.
_DP_CONCENTRATION_UNITS = ('wt%', 'ppm', 'ppb')

def _dp_detect_unit(original_header: str):
    """Extract a concentration unit from a raw column header, if present.

    Recognised units (case-insensitive) map to the canonical strings
    ``'wt%'`` / ``'ppm'`` / ``'ppb'``.  Returns ``None`` if nothing is
    detected — callers fall back to the registry default for the picked
    display label.

    Examples
    --------
    >>> _dp_detect_unit('SiO2 (wt%)')   # 'wt%'
    >>> _dp_detect_unit('SIO2wt')       # 'wt%' (trailing 'wt' suffix)
    >>> _dp_detect_unit('La (ppm)')     # 'ppm'
    >>> _dp_detect_unit('Yb_ppb')       # 'ppb'
    >>> _dp_detect_unit('Sr')           # None
    """
    if not original_header:
        return None
    s = str(original_header).lower()
    # 1) Tokenise on common separators + brackets and look for unit tokens.
    s_norm = re.sub(r'[\[\]()_\-/]+', ' ', s)
    s_norm = re.sub(r'\s+', ' ', s_norm).strip()
    tokens = set(s_norm.split())
    if 'ppb' in tokens:
        return 'ppb'
    if 'ppm' in tokens:
        return 'ppm'
    if any(tok in tokens for tok in ('wt%', 'wtpct', 'wtpercent', 'percent', 'pct')):
        return 'wt%'
    if '%' in s:
        return 'wt%'
    # 2) No-separator suffix form (e.g. 'SIO2wt', 'CRwtpct', 'NIppm') — strip
    # all non-alphanumerics and check the trailing suffix.
    compact = re.sub(r'[^a-z0-9]+', '', s)
    if compact.endswith('ppb'):
        return 'ppb'
    if compact.endswith('ppm'):
        return 'ppm'
    for suff in ('wtpct', 'wtpercent', 'wt'):
        if compact.endswith(suff):
            return 'wt%'
    return None


def _dp_unit_scale(source_unit, target_unit):
    """Multiplicative factor to convert *source_unit* → *target_unit*.

    Returns 1.0 when either unit is None / not a recognised concentration
    unit, or when source equals target — leaves the value untouched.
    """
    if source_unit == target_unit or source_unit is None or target_unit is None:
        return 1.0
    s = _DP_UNIT_TO_PPM.get(source_unit)
    t = _DP_UNIT_TO_PPM.get(target_unit)
    if s is None or t is None:
        return 1.0
    return s / t

# {display_label: (internal_name, default_unit, conv)}
_DP_DISPLAY_TO_INTERNAL = {dl: (iname, unit, conv)
                           for dl, iname, _cat, unit, conv in _DP_REGISTRY}
# {internal_name: display_label}  — prefers conv=None entries
_DP_INTERNAL_TO_DISPLAY: dict = {}
for _dl, _iname, _cat, _unit, _conv in _DP_REGISTRY:
    if _iname not in _DP_INTERNAL_TO_DISPLAY and _conv is None:
        _DP_INTERNAL_TO_DISPLAY[_iname] = _dl
# {display_label: default_unit} — used by the editor when populating the
# Unit column for a freshly auto-mapped row that has no detected unit.
_DP_DISPLAY_TO_DEFAULT_UNIT = {dl: unit for dl, _i, _c, unit, _cv in _DP_REGISTRY}

_DP_KEEP_ORIGINAL = '— keep original —'
_DP_DISPLAY_LABELS = [_DP_KEEP_ORIGINAL] + [dl for dl, _, _, _, _ in _DP_REGISTRY]

# Column-type classification helpers
# {display_label: registry_category}
_DP_DISPLAY_TO_REGCAT: dict = {dl: cat for dl, _iname, cat, _unit, _conv in _DP_REGISTRY}
# Registry category → user-visible column type
_DP_REG_CAT_TO_TYPE: dict = {
    'meta':     'Metadata',
    'location': 'Numeric',
    'age':      'Numeric',
    'class':    'Category',
    'target':   'Numeric',
    'major':    'Numeric',
    'ree':      'Numeric',
    'trace':    'Numeric',
}

# ── Export column labels: internal_name → human-readable header ───────────────
_EXPORT_COL_LABELS: dict = {
    **{iname: dl for dl, iname, _cat, _unit, _cv in _DP_REGISTRY},   # elements/registry
    # Override / extend with result-specific names
    'Sample_ID': 'Sample ID', 'Dataset': 'Dataset', 'Reference': 'Reference',
    'Notes': 'Notes', 'Analytical_Total': 'Analytical total [wt%]',
    'Lat': 'Lat [DD]', 'Lon': 'Lon [DD]', 'Elevation_km': 'Elevation [km]',
    'Lat_Min': 'Lat min [DD]', 'Lat_Max': 'Lat max [DD]',
    'Lon_Min': 'Lon min [DD]', 'Lon_Max': 'Lon max [DD]',
    'Age_Ma': 'Age [Ma]', 'Age_Min_Ma': 'Age min [Ma]', 'Age_Max_Ma': 'Age max [Ma]',
    'Geologic_Era': 'Era', 'Geologic_Period': 'Period',
    'Geologic_Epoch': 'Epoch', 'Geologic_Stage': 'Stage',
    'Geologic_Age_Label': 'Geologic age',
    'Rock_Type': 'Lithology text', 'Rock_Type_Model': 'Rock class',
    'Tectonic_Setting': 'Tectonic setting',
    'Arc_or_Segment': 'Arc / segment', 'Arc': 'Arc', 'Segment': 'Segment',
    'Geologic_Domain': 'Geologic domain',
    'Belt': 'Belt', 'Sub_Belt': 'Sub-belt',
    'Country': 'Country', 'Location': 'Location',
    'Lithology_Type': 'Lithology', 'Lithology_Grouping': 'Lithology group',
    # Predictions & uncertainty
    'Predicted_km': 'Predicted H [km]',
    'Predicted_CI90_Low_km': 'CI90 low [km]',
    'Predicted_CI90_High_km': 'CI90 high [km]',
    'Predicted_CI90_Width_km': 'CI90 width [km]',
    'Crust_Thickness': 'Observed H [km]',
    'Residual_km': 'Residual [km]',
    'Observed_km': 'Observed H [km]',
    'Model': 'Model', 'Algorithm': 'Algorithm',
    'Group_ID': 'Group ID', 'Grouping_Method': 'Grouping method',
    # CRUST1
    'CRUST1_Total_Crust_km': 'CRUST1 total [km]',
    'CRUST1_Crystalline_Crust_km': 'CRUST1 crystalline [km]',
    'CRUST1_Sediment_km': 'CRUST1 sediment [km]',
    'CRUST1_Match_Distance_km': 'CRUST1 distance [km]',
    'CRUST1_Vp': 'CRUST1 Vp', 'CRUST1_Density': 'CRUST1 density',
    # Ratios
    'Sr_Y': 'Sr/Y', 'La_Yb_raw': 'La/Yb', 'La_Yb_N': 'La/YbN',
    'Ce_Y': 'Ce/Y', 'Zr_Y': 'Zr/Y', 'Dy_Yb': 'Dy/Yb',
    'Gd_Yb': 'Gd/Yb', 'Sm_Yb': 'Sm/Yb', 'Ce_Yb': 'Ce/Yb',
    'Nd_Yb': 'Nd/Yb', 'Nb_Yb': 'Nb/Yb', 'Th_Yb': 'Th/Yb',
    'La_Y': 'La/Y', 'Nd_Y': 'Nd/Y', 'Nb_Y': 'Nb/Y', 'Th_Y': 'Th/Y',
    'La_Sm': 'La/Sm', 'Ba_V': 'Ba/V', 'Ba_Sc': 'Ba/Sc',
    'Ni_Sc': 'Ni/Sc', 'Ni_V': 'Ni/V', 'Cr_Sc': 'Cr/Sc',
    'Cr_V': 'Cr/V', 'Lu_Hf': 'Lu/Hf', 'Zr_Ti': 'Zr/Ti', 'Rb_Sr': 'Rb/Sr',
    # Local estimates
    'Local_Median_H_km': 'Local median H [km]', 'Local_Mean_H_km': 'Local mean H [km]',
    'Local_SD_H_km': 'Local SD H [km]', 'Local_N': 'Local N',
    'Local_Radius_km': 'Local radius [km]', 'Good_Local_Estimate': 'Good estimate',
    'Bootstrap_95CI_Low_Median': 'Bootstrap CI low [km]',
    'Bootstrap_95CI_High_Median': 'Bootstrap CI high [km]',
}

# ── Export column groups for shortcut buttons ──────────────────────────────────
_EXPORT_COL_GROUPS: dict = {
    'identity':    ['Sample_ID', 'Dataset', 'Reference', 'Notes'],
    'location':    ['Lat', 'Lon', 'Elevation_km', 'Country', 'Location',
                    'Lat_Min', 'Lat_Max', 'Lon_Min', 'Lon_Max'],
    'age':         ['Age_Ma', 'Age_Min_Ma', 'Age_Max_Ma', 'Geologic_Era',
                    'Geologic_Period', 'Geologic_Epoch', 'Geologic_Stage',
                    'Geologic_Age_Label'],
    'class':       ['Rock_Type', 'Rock_Type_Model', 'Tectonic_Setting',
                    'Arc_or_Segment', 'Arc', 'Segment', 'Geologic_Domain',
                    'Belt', 'Sub_Belt', 'Lithology_Type', 'Lithology_Grouping'],
    'major':       ['SiO2','TiO2','Al2O3','FeO','Fe2O3','Fe2O3T','MnO','MgO',
                    'CaO','Na2O','K2O','P2O5','LOI','H2Ot'],
    'ree':         ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'],
    'trace':       ['Rb','Sr','Y','Zr','Nb','Ba','Hf','Ta','Pb','Th','U',
                    'Sc','V','Cr','Co','Ni','Cu','Zn','Ga','Li','Be','Cs',
                    'Bi','Sn','W','Mo','Ge'],
    'ratio':       ['Sr_Y','La_Yb_raw','La_Yb_N','Ce_Y','Zr_Y','Dy_Yb','Gd_Yb',
                    'Sm_Yb','Ce_Yb','Nd_Yb','Nb_Yb','Th_Yb','La_Y','Nd_Y',
                    'Nb_Y','Th_Y','La_Sm','Ba_V','Ba_Sc','Ni_Sc','Ni_V',
                    'Cr_Sc','Cr_V','Lu_Hf','Zr_Ti','Rb_Sr'],
    'prediction':  ['Predicted_km','Predicted_CI90_Low_km','Predicted_CI90_High_km',
                    'Predicted_CI90_Width_km','Model','Algorithm'],
    'grouping':    ['Group_ID','Grouping_Method'],
    'validation':  ['Crust_Thickness','Observed_km','Residual_km',
                    'CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km',
                    'CRUST1_Sediment_km','CRUST1_Match_Distance_km'],
}


def _export_label(col: str) -> str:
    """Human-readable label for a column, used in the export column selector."""
    return _EXPORT_COL_LABELS.get(col, col.replace('_', ' '))


def _build_export_excel(sheets: dict) -> bytes:
    """Build a multi-sheet Excel workbook. sheets = {sheet_name: DataFrame}."""
    import io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    import math

    hdr_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    hdr_font = Font(bold=True, color='FFFFFF')

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for raw_name, df in sheets.items():
        ws = wb.create_sheet(raw_name[:31])
        # Rename to human-readable headers
        display_cols = [_export_label(c) for c in df.columns]
        # Write header row
        for ci, hdr in enumerate(display_cols, 1):
            cell = ws.cell(row=1, column=ci, value=hdr)
            cell.fill = hdr_fill; cell.font = hdr_font
            cell.alignment = Alignment(horizontal='center')
        # Write data — replace NaN/inf with None (Excel blank)
        for ri, row_vals in enumerate(df.values.tolist(), 2):
            for ci, val in enumerate(row_vals, 1):
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    val = None
                ws.cell(row=ri, column=ci, value=val)
        # Column widths
        for ci, (hdr, icol) in enumerate(zip(display_cols, df.columns), 1):
            sample_max = df[icol].astype(str).str.len().quantile(0.9) if len(df) else 0
            width = min(max(len(hdr) + 2, int(sample_max) + 1), 26)
            ws.column_dimensions[get_column_letter(ci)].width = width
        ws.freeze_panes = 'A2'
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _dp_mapping_to_excel(orig_cols: list, cur_map: dict, confirmed: set) -> bytes | None:
    """Return bytes of an xlsx mapping workbook with validated dropdowns for each row."""
    import io
    try:
        import openpyxl
        from openpyxl.worksheet.datavalidation import DataValidation
        from openpyxl.styles import PatternFill, Font, Alignment, Protection
    except Exception:
        return None

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Mapping'

    # Hidden sheet listing all valid labels (Excel list validation needs a sheet reference
    # when the list exceeds ~255 chars in a formula string)
    ws_lbl = wb.create_sheet('_Labels')
    for i, lbl in enumerate(_DP_DISPLAY_LABELS, 1):
        ws_lbl.cell(row=i, column=1).value = lbl
    ws_lbl.sheet_state = 'hidden'

    # ── Header row ──────────────────────────────────────────────────────────────
    hdr_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    hdr_font = Font(bold=True, color='FFFFFF')
    for col_idx, title in enumerate(['Original header', 'Map to', 'Confirmed'], 1):
        cell = ws.cell(row=1, column=col_idx, value=title)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal='center')

    # ── Data rows ───────────────────────────────────────────────────────────────
    n = len(orig_cols)
    for row_idx, orig in enumerate(orig_cols, 2):
        ws.cell(row=row_idx, column=1, value=orig)
        ws.cell(row=row_idx, column=2, value=cur_map.get(orig, _DP_KEEP_ORIGINAL))
        ws.cell(row=row_idx, column=3, value='Yes' if orig in confirmed else 'No')

    # ── Data validation: Map to column uses the _Labels sheet ───────────────────
    dv_map = DataValidation(
        type='list',
        formula1=f"'_Labels'!$A$1:$A${len(_DP_DISPLAY_LABELS)}",
        allow_blank=False,
        showDropDown=False,
        showErrorMessage=True,
        error='Select a value from the dropdown list.',
        errorTitle='Invalid mapping',
    )
    ws.add_data_validation(dv_map)
    dv_map.sqref = f'B2:B{n + 1}'

    # ── Data validation: Confirmed column ───────────────────────────────────────
    dv_conf = DataValidation(type='list', formula1='"Yes,No"', allow_blank=False,
                             showDropDown=False)
    ws.add_data_validation(dv_conf)
    dv_conf.sqref = f'C2:C{n + 1}'

    # ── Sheet protection: column A read-only, B and C editable ──────────────────
    ws.protection.sheet = True
    ws.protection.password = ''
    ws.protection.enable()
    for row in ws.iter_rows(min_row=2, max_row=n + 1, min_col=1, max_col=1):
        for cell in row:
            cell.protection = Protection(locked=True)
    for row in ws.iter_rows(min_row=2, max_row=n + 1, min_col=2, max_col=3):
        for cell in row:
            cell.protection = Protection(locked=False)

    ws.column_dimensions['A'].width = 32
    ws.column_dimensions['B'].width = 32
    ws.column_dimensions['C'].width = 12
    ws.freeze_panes = 'A2'

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _dp_excel_to_mapping(file, orig_cols: list, cur_map: dict, confirmed: set):
    """Parse an xlsx mapping workbook and return (new_map, new_confirmed) merged with current."""
    try:
        import openpyxl
        file.seek(0)
        wb = openpyxl.load_workbook(file, data_only=True, read_only=True)
        ws = wb['Mapping']
        xl_map: dict = {}
        xl_conf: set = set()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] is None:
                continue
            orig = str(row[0]).strip()
            mapped = str(row[1]).strip() if row[1] is not None else _DP_KEEP_ORIGINAL
            conf_val = str(row[2]).strip().lower() if row[2] is not None else 'no'
            xl_map[orig] = mapped if mapped in _DP_DISPLAY_LABELS else _DP_KEEP_ORIGINAL
            if conf_val in ('yes', 'true', '1'):
                xl_conf.add(orig)
        wb.close()
        new_map = {c: xl_map.get(c, cur_map.get(c, _DP_KEEP_ORIGINAL)) for c in orig_cols}
        new_conf = {c for c in orig_cols if c in xl_conf or
                    (c not in xl_map and c in confirmed)}
        return new_map, new_conf
    except KeyError as err:
        raise ValueError("Excel file must contain a sheet named 'Mapping'.") from err
    except Exception as exc:
        raise ValueError(str(exc)) from exc


# ── Persistent user-alias learning ────────────────────────────────────────────
_DP_USER_ALIAS_FILE = _APP_DIR / '_dp_user_aliases.json'

@st.cache_data(show_spinner=False)
def _dp_load_user_aliases_cached(mtime: float) -> dict:
    """Cached body of _dp_load_user_aliases — keyed on file mtime so an
    alias save (which bumps mtime) automatically invalidates the cache."""
    try:
        if _DP_USER_ALIAS_FILE.exists():
            data = json.loads(_DP_USER_ALIAS_FILE.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                return {}
            # Migrate: strip trailing ' [wt%]' / ' [ppm]' / ' [ppb]' / ' [DD]' /
            # ' [m]' / ' [km]' / ' [Ma]' suffix from values.
            migrated: dict = {}
            for k, v in data.items():
                if isinstance(v, str):
                    new_v = re.sub(r'\s*\[[^\]]+\]\s*$', '', v)
                    # Drop the entry if migration produced a label that's
                    # no longer in the registry (e.g. old typo, removed entry).
                    if new_v in _DP_DISPLAY_TO_INTERNAL or new_v == _DP_KEEP_ORIGINAL:
                        migrated[k] = new_v
                    elif v in _DP_DISPLAY_TO_INTERNAL:
                        # Original (unmigrated) value is still valid — keep it.
                        migrated[k] = v
                    # else: stale entry, skip silently
                else:
                    migrated[k] = v
            return migrated
    except Exception:
        pass
    return {}

def _dp_load_user_aliases() -> dict:
    """Load saved column-name aliases, migrating pre-bare-label entries.

    Older versions of the app stored display labels with unit suffixes
    (`'FeO [wt%]'`, `'La [ppm]'`, `'La [ppb]'`).  The current registry uses
    bare names so we strip the suffix on load.  ``[ppb]`` entries are kept
    as the bare element name — the per-column Unit field handles the
    ppb→ppm conversion now (so `'sr ppb' → 'Sr [ppb]'` becomes `'sr ppb' →
    'Sr'`, and the auto-mapper picks unit='ppb' from the original header).
    Result is cached per file-mtime so saving a new alias automatically
    invalidates; otherwise the on-disk read happens once per session.
    """
    if not _DP_USER_ALIAS_FILE.exists():
        return {}
    try:
        _mtime = _DP_USER_ALIAS_FILE.stat().st_mtime
    except Exception:
        return {}
    return _dp_load_user_aliases_cached(_mtime)

def _dp_save_user_alias(orig_col: str, display_label: str) -> None:
    """Persist a user-corrected column mapping keyed by normalised header."""
    try:
        aliases = _dp_load_user_aliases()
        k = key(orig_col)
        if display_label == _DP_KEEP_ORIGINAL:
            aliases.pop(k, None)
        else:
            aliases[k] = display_label
        _DP_USER_ALIAS_FILE.write_text(json.dumps(aliases, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass


# ── Default reference-dataset preload list ──────────────────────────────────
# Persistent list of built-in reference datasets the user wants the Prepare
# tab to auto-load on every session. Populated by the "💾 Save as default"
# button in the reference loader expander; consumed when ``_dp_ref_loaded``
# is first initialised in session state. Column-mapping decisions are saved
# separately via ``_dp_save_user_alias`` (per-column-name), so this file only
# needs to record the *which references to load*, not the mappings themselves.
_DP_DEFAULT_REFS_FILE = _APP_DIR / '_dp_default_refs.json'

def _dp_load_default_refs() -> list:
    """Return the saved list of reference-dataset display labels, or []."""
    try:
        if _DP_DEFAULT_REFS_FILE.exists():
            data = json.loads(_DP_DEFAULT_REFS_FILE.read_text(encoding='utf-8'))
            if isinstance(data, list):
                return [str(x) for x in data]
    except Exception:
        pass
    return []

def _dp_save_default_refs(labels: list) -> bool:
    """Overwrite the default-refs JSON with *labels*. Returns True on success."""
    try:
        _DP_DEFAULT_REFS_FILE.write_text(
            json.dumps(list(labels), indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        return True
    except Exception:
        return False


# ── Custom (user-registered) reference files ────────────────────────────────
# The Prepare-tab reference loader's three built-in buttons (Guo, Zou, Luffi)
# can be supplemented with arbitrary user-pinned files: friendly_name → file
# path on disk. These appear alongside the built-ins as one-click load
# buttons, and can be added to the default preload list the same way.
#
# Stored as a JSON list of {'label': str, 'path': str} dicts.
_DP_CUSTOM_REFS_FILE = _APP_DIR / '_dp_custom_refs.json'

def _dp_load_custom_refs() -> list:
    """Return the list of registered custom reference files.

    Each entry is a dict with keys 'label' and 'path'. Bad entries (missing
    keys, non-string types) are silently filtered out.
    """
    try:
        if _DP_CUSTOM_REFS_FILE.exists():
            data = json.loads(_DP_CUSTOM_REFS_FILE.read_text(encoding='utf-8'))
            if isinstance(data, list):
                cleaned = []
                for entry in data:
                    if (isinstance(entry, dict)
                            and isinstance(entry.get('label'), str)
                            and isinstance(entry.get('path'), str)
                            and entry['label'].strip()
                            and entry['path'].strip()):
                        cleaned.append({
                            'label': entry['label'].strip(),
                            'path':  entry['path'].strip(),
                        })
                return cleaned
    except Exception:
        pass
    return []

def _dp_save_custom_refs(entries: list) -> bool:
    """Overwrite the custom-refs JSON with *entries*. Returns True on success."""
    try:
        _DP_CUSTOM_REFS_FILE.write_text(
            json.dumps(list(entries), indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        return True
    except Exception:
        return False

# ── Geologic time → Age [Ma] lookup ───────────────────────────────────────────
_DP_GEOLOGIC_AGE_MA: dict = {
    # Eons / supereons
    'hadean':3975, 'archean':2750, 'proterozoic':1500, 'phanerozoic':280,
    'precambrian':2000,
    # Eras
    'paleozoic':395, 'mesozoic':185, 'cenozoic':33,
    # Periods
    'cambrian':510, 'ordovician':465, 'silurian':433, 'devonian':385,
    'carboniferous':325, 'mississippian':340, 'pennsylvanian':307,
    'permian':275, 'triassic':232, 'jurassic':165, 'cretaceous':96,
    'paleogene':55, 'neogene':12, 'quaternary':1.5,
    # Epochs
    'paleocene':59, 'eocene':46, 'oligocene':31,
    'miocene':14, 'pliocene':3.5, 'pleistocene':1.0, 'holocene':0.01,
    # Common stages
    'campanian':80, 'maastrichtian':69, 'turonian':91, 'cenomanian':97,
    'albian':106, 'aptian':120, 'barremian':127, 'hauterivian':131,
    'valanginian':136, 'berriasian':141, 'tithonian':149, 'kimmeridgian':155,
    'oxfordian':160, 'callovian':165, 'bathonian':169, 'bajocian':172,
    'aalenian':174, 'toarcian':181, 'pliensbachian':189, 'sinemurian':195,
    'hettangian':202, 'rhaetian':205, 'norian':220, 'carnian':235,
    'ladinian':242, 'anisian':247, 'olenekian':250, 'induan':252,
    'changhsingian':254, 'wuchiapingian':257, 'capitanian':262,
    'wordian':265, 'roadian':268, 'kungurian':277, 'artinskian':285,
    'sakmarian':295, 'asselian':299, 'gzhelian':302, 'kasimovian':305,
    'moscovian':311, 'bashkirian':319, 'serpukhovian':327,
    'visean':335, 'tournaisian':350,
    'famennian':366, 'frasnian':374, 'givetian':382, 'eifelian':388,
    'emsian':400, 'pragian':408, 'lochkovian':413,
    'pridoli':424, 'ludlow':427, 'wenlock':432, 'llandovery':440,
    'hirnantian':446, 'katian':454, 'sandbian':457, 'darriwilian':462,
    'dapingian':468, 'floian':474, 'tremadocian':481,
    'furongian':494, 'miaolingian':509, 'series2':521, 'terreneuvian':534,
}

def _dp_infer_age_from_geologic_time(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Age_Ma from Geologic_Epoch/Stage/Period/Era/Age where absent."""
    age_src_cols = [c for c in ['Geologic_Stage','Geologic_Epoch','Geologic_Period',
                                'Geologic_Era','Geologic_Age'] if c in df.columns]
    if not age_src_cols:
        return df
    if 'Age_Ma' not in df.columns:
        df = df.copy(); df['Age_Ma'] = np.nan
    mask = df['Age_Ma'].isna()
    if not mask.any():
        return df
    df = df.copy()
    for col in age_src_cols:
        still_missing = df['Age_Ma'].isna()
        if not still_missing.any():
            break
        vals = df.loc[still_missing, col].astype(str).str.strip().str.lower()
        inferred = vals.map(_DP_GEOLOGIC_AGE_MA)
        df.loc[still_missing & inferred.notna(), 'Age_Ma'] = inferred[inferred.notna()]
    return df


# Ratios computed automatically from element columns — not exposed in the mapper dropdown
_DP_AUTO_RATIOS: dict = {
    'Sr_Y':     ('Sr',  'Y'),
    'La_Yb_raw':('La',  'Yb'),
    'Ce_Y':     ('Ce',  'Y'),
    'Zr_Y':     ('Zr',  'Y'),
    'Dy_Yb':    ('Dy',  'Yb'),
    'Gd_Yb':    ('Gd',  'Yb'),
    'Sm_Yb':    ('Sm',  'Yb'),
    'Ce_Yb':    ('Ce',  'Yb'),
    'Nd_Yb':    ('Nd',  'Yb'),
    'Nb_Yb':    ('Nb',  'Yb'),
    'Th_Yb':    ('Th',  'Yb'),
    'La_Y':     ('La',  'Y'),
    'Nd_Y':     ('Nd',  'Y'),
    'Nb_Y':     ('Nb',  'Y'),
    'Th_Y':     ('Th',  'Y'),
    'La_Sm':    ('La',  'Sm'),
    'Ba_V':     ('Ba',  'V'),
    'Ba_Sc':    ('Ba',  'Sc'),
    'Ni_Sc':    ('Ni',  'Sc'),
    'Ni_V':     ('Ni',  'V'),
    'Cr_Sc':    ('Cr',  'Sc'),
    'Cr_V':     ('Cr',  'V'),
    'Lu_Hf':    ('Lu',  'Hf'),
    'Zr_Ti':    ('Zr',  'Ti'),
    'Rb_Sr':    ('Rb',  'Sr'),
}
_DP_CHON_LA, _DP_CHON_YB = 0.237, 0.161   # CI chondrite [ppm] — McDonough & Sun 1995


def _dp_null_below_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Replace negative concentrations in measurement columns with NaN.

    Negative values in geochemical data are always below-detection-limit (BDL)
    artefacts — they are unphysical concentrations.  Setting them to NaN is the
    correct approach; using half the detection limit (DL/2) is discouraged,
    especially before computing ratios where a BDL numerator or denominator
    would produce meaningless negative or near-zero ratio values.

    Applied to all registry measurement columns (major oxides, REE, trace
    elements, target, age) that are present in the DataFrame, immediately
    before ratio computation so ratios inherit clean NaN values.
    """
    # _DP_QA_MEASUREMENT_COLS is defined later in the file but Python resolves
    # names at call time, not definition time, so this is safe.
    result = df.copy()
    for col in _DP_QA_MEASUREMENT_COLS:  # noqa: F821  (defined below at module level)
        if col not in result.columns:
            continue
        ser = pd.to_numeric(result[col], errors='coerce')
        # Any negative value is BDL → NaN.  Zero is kept (valid for trace elements
        # at the detection floor, though it will be excluded from log-ratio plots).
        result[col] = ser.where(ser >= 0, other=np.nan)
    return result


def _dp_compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all standard geochemical ratios from element columns already present."""
    out = df
    for ratio_col, (num_el, den_el) in _DP_AUTO_RATIOS.items():
        if ratio_col not in out.columns and num_el in out.columns and den_el in out.columns:
            n = pd.to_numeric(out[num_el], errors='coerce')
            d = pd.to_numeric(out[den_el], errors='coerce')
            out = out.copy() if out is df else out
            out[ratio_col] = np.where(d.notna() & (d != 0), n / d, np.nan)
    if 'La_Yb_N' not in out.columns and 'La' in out.columns and 'Yb' in out.columns:
        la_n = pd.to_numeric(out['La'], errors='coerce') / _DP_CHON_LA
        yb_n = pd.to_numeric(out['Yb'], errors='coerce') / _DP_CHON_YB
        out = out.copy() if out is df else out
        out['La_Yb_N'] = np.where(yb_n.notna() & (yb_n != 0), la_n / yb_n, np.nan)
    return out


# Major oxides included in anhydrous normalisation (volatiles excluded).
# After _dp_apply_mapping, iron is always resolved to 'FeO' (= FeOt equivalent);
# Fe2O3 and Fe2O3T are consumed during mapping and will not be in the DataFrame.
_DP_MAJOR_OXIDE_ANHY = ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5']


def _dp_anhydrous_recalc(df: pd.DataFrame) -> pd.DataFrame:
    """Renormalise major oxides to 100 % anhydrous.

    Each oxide = oxide / sum(anhydrous oxides) * 100, rounded to 2 d.p.
    LOI and H2Ot are excluded from the sum (they are volatiles, not structural
    oxides) so the formula is identical to the user-facing description:
      SiO2_anhy = SiO2_raw / (analytical_total − LOI) * 100
    This is the standard preparation for thermodynamic modelling and most
    petrogenetic discrimination diagrams.
    """
    out = df.copy()
    ox = [c for c in _DP_MAJOR_OXIDE_ANHY if c in out.columns]
    if not ox:
        return out
    num = out[ox].apply(pd.to_numeric, errors='coerce')
    # Require at least half the oxides to be present for a valid sum
    total = num.sum(axis=1, min_count=max(1, len(ox) // 2))
    for c in ox:
        out[c] = np.where(total.notna() & (total > 0), (num[c] / total * 100).round(2), out[c])
    return out


def _dp_major_qc_panel(df: pd.DataFrame, key_prefix: str) -> None:
    """LOI > 2 % and analytical total outside 98–102 % QC flags.

    Shows warnings and a table of flagged rows. Rows can be individually
    ignored (excluded from downstream tabs) using the Ignore checkbox.
    """
    _has_major = any(c in df.columns for c in _DP_MAJOR_OXIDE_ANHY)
    if not _has_major:
        st.info('Map major element columns to enable QC checks.')
        return

    _ignore_key = f'{key_prefix}_ignore'
    _has_loi    = 'LOI' in df.columns

    # ── Compute analytical totals with correct iron and volatile handling ───
    # Iron priority: Fe2O3T = total iron → use exclusively.
    #   Else FeO + Fe2O3 (speciated pair, add both).
    #   Else whichever single iron column is present.
    # Volatile priority: LOI supersedes H2Ot (H2Ot is a subset of LOI —
    #   adding both would double-count water).
    def _safe(col):
        return pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(0.0, index=df.index)

    # After _dp_apply_mapping, 'FeO' always holds FeOt (total iron as FeO
    # equivalent); Fe2O3 / Fe2O3T are consumed and dropped during mapping.
    # H2Ot is a component of LOI — never add both.
    _base_ox = [c for c in ['SiO2','TiO2','Al2O3','MnO','MgO','CaO','Na2O','K2O','P2O5'] if c in df.columns]
    _base_sum = df[_base_ox].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1) if _base_ox else pd.Series(0.0, index=df.index)
    _iron = _safe('FeO')   # always FeOt after mapping resolution
    _vol  = _safe('LOI') if 'LOI' in df.columns else (_safe('H2Ot') if 'H2Ot' in df.columns else pd.Series(0.0, index=df.index))

    _tots_raw = _base_sum + _iron + _vol
    _n_ox_present = len(_base_ox) + (1 if 'FeO' in df.columns else 0)
    _tots = _tots_raw.where(_base_sum.notna() & (_n_ox_present >= 4)).round(2)

    _loi_vals = pd.to_numeric(df['LOI'], errors='coerce') if _has_loi else pd.Series(np.nan, index=df.index)

    _loi_flag = _loi_vals > 2.0
    _tot_flag = (_tots < 98.0) | (_tots > 102.0)
    _any_flag = _loi_flag | _tot_flag
    _n_all    = len(df)

    n_loi = int(_loi_flag.sum())
    n_tot = int(_tot_flag.sum())

    if n_loi > 0:
        st.warning(
            f'⚠ **{n_loi} of {_n_all}** samples have LOI > 2 wt% — '
            'Ca, Na and LILE may be partially mobilised.')
    if n_tot > 0:
        st.warning(
            f'⚠ **{n_tot} of {_n_all}** samples have analytical total outside '
            '98–102 % — check for missing oxides or transcription errors.')
    if n_loi == 0 and n_tot == 0:
        st.success('✓ All samples pass LOI and analytical-total checks.')

    # ── Flagged rows table with optional per-row ignore ─────────────────────
    _id_col = 'Sample_ID' if 'Sample_ID' in df.columns else None
    _ignore_set = st.session_state.get(_ignore_key, set())

    if _any_flag.any():
        _flag_rows = df[_any_flag].copy().reset_index()
        _flag_df = pd.DataFrame({
            'Ignore': (_flag_rows['Sample_ID'].astype(str).isin({str(s) for s in _ignore_set})
                       if _id_col else [i in _ignore_set for i in _flag_rows['index']]),
            'Sample_ID': _flag_rows[_id_col].astype(str) if _id_col else _flag_rows['index'].astype(str),
            'Analytical total': _tots[_any_flag].values,
            'LOI [wt%]': _loi_vals[_any_flag].round(2).values if _has_loi else np.nan,
            'LOI > 2%': _loi_flag[_any_flag].values,
            'Total <98 or >102': _tot_flag[_any_flag].values,
        })
        st.caption(f'{len(_flag_df)} flagged row(s) — tick **Ignore** to exclude from downstream tabs:')
        _cc = {
            'Ignore':           st.column_config.CheckboxColumn('Ignore', default=False),
            'Sample_ID':        st.column_config.TextColumn('Sample ID', disabled=True),
            'Analytical total': st.column_config.NumberColumn('Total', disabled=True, format='%.2f'),
            'LOI [wt%]':        st.column_config.NumberColumn('LOI [wt%]', disabled=True, format='%.2f'),
            'LOI > 2%':         st.column_config.CheckboxColumn('LOI>2%', disabled=True),
            'Total <98 or >102':st.column_config.CheckboxColumn('Total<98/>102', disabled=True),
        }
        _qc_edited = st.data_editor(
            _flag_df, column_config=_cc,
            disabled=[c for c in _flag_df.columns if c != 'Ignore'],
            hide_index=True, use_container_width=True,
            key=f'{key_prefix}_qc_editor',
        )
        _new_ignore = set(_qc_edited.loc[_qc_edited['Ignore'], 'Sample_ID'].astype(str))
        if _new_ignore != _ignore_set:
            st.session_state[_ignore_key] = _new_ignore
            st.rerun()
        if _new_ignore:
            st.caption(f'{len(_new_ignore)} row(s) ignored and excluded from downstream tabs.')


def _dp_utm_to_dd(easting, northing, zone_number, northern=True):
    """Convert UTM arrays to WGS84 decimal degrees (pure numpy, WGS84 ellipsoid)."""
    a = 6378137.0; f = 1.0/298.257223563
    e2 = 2*f - f**2; ep2 = e2/(1-e2); k0 = 0.9996
    x = np.asarray(easting, float) - 500000.0
    y = np.asarray(northing, float)
    if not northern:
        y = y - 10000000.0
    lon0 = np.radians((zone_number - 1)*6 - 180 + 3)
    M = y/k0
    mu = M/(a*(1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1 = (1 - np.sqrt(1-e2))/(1 + np.sqrt(1-e2))
    phi1 = (mu
            + (3*e1/2    - 27*e1**3/32) *np.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32)*np.sin(4*mu)
            + (151*e1**3/96)             *np.sin(6*mu)
            + (1097*e1**4/512)           *np.sin(8*mu))
    N1 = a/np.sqrt(1 - e2*np.sin(phi1)**2)
    T1 = np.tan(phi1)**2
    C1 = ep2*np.cos(phi1)**2
    R1 = a*(1-e2)/(1 - e2*np.sin(phi1)**2)**1.5
    D = x/(N1*k0)
    lat = phi1 - (N1*np.tan(phi1)/R1)*(
        D**2/2
        - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*ep2)*D**4/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*ep2 - 3*C1**2)*D**6/720)
    lon = lon0 + (D - (1 + 2*T1 + C1)*D**3/6
                  + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*ep2 + 24*T1**2)*D**5/120)/np.cos(phi1)
    return np.degrees(lat), np.degrees(lon)


def _dp_excel_sheet_names(file) -> list[str]:
    """Return sheet names for an Excel file without consuming the stream.

    Wraps ``_dp_excel_sheet_names_cached`` to avoid re-parsing a 10-30 MB
    Excel file on every Streamlit rerun (every lasso, every button click).
    The cache key is ``(file.name, file.size)`` — stable across reruns for
    the same upload, and underscore-prefixed bytes are excluded from the
    hash so Streamlit doesn't have to re-hash 17 MB on every call.
    """
    try:
        _name = getattr(file, 'name', '')
        _size = getattr(file, 'size', None)
        if _size is None:
            try:
                file.seek(0, 2)   # SEEK_END
                _size = file.tell()
                file.seek(0)
            except Exception:
                _size = -1
        file.seek(0)
        _bytes = file.read()
        file.seek(0)
        return _dp_excel_sheet_names_cached(_name, int(_size), _bytes)
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _dp_excel_sheet_names_cached(file_name: str, file_size: int, _file_bytes: bytes) -> list[str]:
    """Cached sheet-name lookup. The ``_file_bytes`` arg is excluded from
    the cache hash (underscore prefix); ``(file_name, file_size)`` IS the
    real key. Edits that change file size invalidate cleanly; same name +
    same size + different content (rare) would hit a stale cache, which is
    a tradeoff we accept for the speed gain."""
    import io
    bio = io.BytesIO(_file_bytes)
    try:
        import openpyxl
        wb = openpyxl.load_workbook(bio, read_only=True, data_only=True)
        names = wb.sheetnames
        wb.close()
        return names
    except Exception:
        try:
            bio.seek(0)
            xl = pd.ExcelFile(bio)
            return xl.sheet_names
        except Exception:
            return []


def _dp_read_raw(file, sheet_name=0):
    """Read file preserving original column names, with header-row detection.
    sheet_name: int index or str name (Excel only); ignored for CSV.

    Wraps ``_dp_read_raw_cached`` so 10-30 MB Excel files (e.g. Luffi T1)
    don't get re-parsed on every Streamlit rerun. Cache key is
    ``(file.name, file.size, sheet_name)`` — stable for the same upload.
    """
    try:
        _name = getattr(file, 'name', str(file))
        _size = getattr(file, 'size', None)
        if _size is None:
            try:
                file.seek(0, 2)
                _size = file.tell()
                file.seek(0)
            except Exception:
                _size = -1
        file.seek(0)
        _bytes = file.read()
        file.seek(0)
        return _dp_read_raw_cached(_name, int(_size), str(sheet_name), _bytes)
    except Exception:
        return pd.DataFrame(), []


@st.cache_data(show_spinner=False, ttl=3600, max_entries=12)
def _dp_read_raw_cached(file_name: str, file_size: int, sheet_name_str: str,
                        _file_bytes: bytes):
    """Cached file-parse. Returns (DataFrame, headers list).

    The ``_file_bytes`` arg is excluded from the hash via the underscore
    prefix; ``(file_name, file_size, sheet_name_str)`` is the cache key.
    See ``_dp_excel_sheet_names_cached`` for the same-name+same-size
    tradeoff rationale.
    """
    import io
    name_lc = file_name.lower()
    bio = io.BytesIO(_file_bytes)
    try:
        if name_lc.endswith(('.xlsx', '.xls')):
            try:
                _sheet_arg = int(sheet_name_str)
            except (ValueError, TypeError):
                _sheet_arg = sheet_name_str if sheet_name_str else 0
            raw = pd.read_excel(bio, header=None, sheet_name=_sheet_arg)
        else:
            raw = pd.read_csv(bio, header=None)
        hdr = None
        for i in range(min(len(raw), 30)):
            if looks_header(raw.iloc[i].tolist()): hdr = i; break
        if hdr is not None:
            headers = [str(v).strip() if pd.notna(v) else f'Col_{j+1}' for j, v in enumerate(raw.iloc[hdr].tolist())]
            df = raw.iloc[hdr+1:].copy(); df.columns = headers
        else:
            headers = [f'Col_{j+1}' for j in range(raw.shape[1])]
            df = raw.copy(); df.columns = headers
        df = df.dropna(how='all').reset_index(drop=True)
        return df, headers
    except Exception:
        return pd.DataFrame(), []


def _dp_auto_map(original_cols):
    """Return ({orig_col: display_label}, {orig_col: unit}).

    The display-label dict is what the editor's "Map to" dropdown reflects.
    The unit dict feeds the editor's "Unit" column and the apply-mapping
    conversion logic — for each mapped column, we record the concentration
    unit detected from the original header (or the registry default when
    no unit is parseable).

    User aliases ( ``_dp_user_aliases.json`` ) take priority over the
    registry. Iron columns are disambiguated FeO vs FeOt by inspecting the
    original column name for a 't' / 'tot' / 'total' marker, since both map
    to internal name ``FeO`` in the registry but produce different paths in
    the iron resolver downstream.
    """
    user_aliases = _dp_load_user_aliases()
    result: dict = {}
    units:  dict = {}
    for c in original_cols:
        k_norm = key(c)
        detected_unit = _dp_detect_unit(c)
        # User-corrected alias takes highest priority
        if k_norm in user_aliases and user_aliases[k_norm] in _DP_DISPLAY_TO_INTERNAL:
            result[c] = user_aliases[k_norm]
            _, default_unit, _ = _DP_DISPLAY_TO_INTERNAL[result[c]]
            units[c] = detected_unit or default_unit
            continue
        iname = canonical_header_name(c)
        if iname is None:
            result[c] = _DP_KEEP_ORIGINAL
            units[c]  = detected_unit
            continue
        # ── Iron disambiguation (FeO vs FeOt) ────────────────────────────
        if iname == 'FeO':
            _ck = compact_key(c)            # all-lower, alphanum only
            _is_total = (
                'feot' in _ck               # 'feotwt', 'feotwt%', 'feotpct'
                or 'fetot' in _ck           # 'fetot', 'fetotal'
                or 'fetotal' in _ck
            )
            result[c] = 'FeOt' if _is_total else 'FeO'
            units[c]  = detected_unit or 'wt%'
            continue
        result[c] = _DP_INTERNAL_TO_DISPLAY.get(iname, _DP_KEEP_ORIGINAL)
        if result[c] != _DP_KEEP_ORIGINAL:
            _, default_unit, _ = _DP_DISPLAY_TO_INTERNAL[result[c]]
            units[c] = detected_unit or default_unit
        else:
            units[c] = detected_unit
    return result, units


def _dp_auto_type(orig_cols, auto_map, raw_df=None):
    """Return {orig_col: 'Numeric'|'Category'|'Metadata'} from registry + data heuristics."""
    result = {}
    for c in orig_cols:
        disp = auto_map.get(c, _DP_KEEP_ORIGINAL)
        if disp != _DP_KEEP_ORIGINAL and disp in _DP_DISPLAY_TO_REGCAT:
            result[c] = _DP_REG_CAT_TO_TYPE.get(_DP_DISPLAY_TO_REGCAT[disp], 'Numeric')
            continue
        # Unmapped column — infer from data
        if raw_df is not None and c in raw_df.columns:
            ser = raw_df[c].dropna()
            if len(ser) == 0:
                result[c] = 'Metadata'
                continue
            numeric_frac = pd.to_numeric(ser, errors='coerce').notna().mean()
            n_uniq = int(ser.nunique())
            n_total = len(ser)
            if numeric_frac >= 0.8:
                result[c] = 'Numeric'
            elif n_uniq <= max(20, int(n_total * 0.1)):
                result[c] = 'Category'
            else:
                result[c] = 'Metadata'
        else:
            result[c] = 'Numeric'
    return result


def _dp_apply_mapping(raw_df, mapping, utm_info=None, unit_map=None,
                      ignore_cols=None, rename_map=None):
    """Apply display_label mapping with unit + element→oxide conversions.

    Parameters
    ----------
    raw_df       : the unprocessed input DataFrame.
    mapping      : ``{orig_col: display_label}`` from the editor.  Bare names
                   from the registry (e.g. ``'SiO2'``, ``'La'``).
    utm_info     : ``(zone_int, is_north_bool)`` if Lat/Lon are coming from UTM.
    unit_map     : ``{orig_col: unit_str}`` (e.g. ``'wt%'`` / ``'ppm'`` / ``'ppb'``).
                   When omitted the registry's default unit for each label is
                   assumed.  Conversion logic:
                     scale = (source_unit → target_unit) × elem_oxide_factor
                   where target_unit is the registry's default for that label,
                   and elem_oxide_factor only fires when the registry entry has
                   ``conv = ('elem_oxide', factor)``.
    ignore_cols  : iterable of original column names the user marked
                   ``Ignore`` in the editor. They are dropped *before* any
                   rename so a duplicate-mapped sibling (e.g. ``SIO2wt`` next
                   to an Ignored ``SiO2 (wt%)``) cleanly gets the rename
                   target name without being collateral-damaged.
    rename_map   : ``{orig_col: custom_target_name}`` overriding the registry's
                   internal_name for individual columns. Lets users keep
                   duplicate-mapped columns under unique custom names — e.g.
                   map both ``SIO2wt`` and ``SiO2 (wt%)`` to display ``SiO2``,
                   then specify rename_map = ``{'SiO2 (wt%)': 'SiO2_anhy'}`` so
                   the model gets clean ``SiO2`` (from SIO2wt) and the
                   recalculated values are preserved under ``SiO2_anhy``. The
                   conversion logic (unit + elem_oxide) still applies; only the
                   final output column name changes.

    Iron resolution
    ───────────────
    Iron can be reported in several ways in the same sheet:
      • FeOt   – total iron already expressed as FeO equivalent (lab-measured)
      • FeO    – speciated ferrous iron only
      • Fe2O3  – speciated ferric iron only
      • Fe2O3t – total iron expressed as Fe2O3 equivalent

    All four are collected separately from the generic column-rename loop and
    resolved into a single ``FeO`` column that always holds FeOt (total iron
    as FeO wt%). Priority:
      1. FeOt directly mapped         → use as-is
      2. Fe2O3t mapped                → × 0.8998
      3. Speciated FeO + Fe2O3 pair   → FeOt = FeO + Fe2O3 × 0.8998
      4. FeO only                     → treat as FeOt (may be lab-reported FeOT)
      5. Fe2O3 only                   → × 0.8998
    Source iron columns are dropped after resolution so they can't inflate
    the analytical total.
    """
    _IRON_LABELS = {'FeOt', 'FeO', 'Fe2O3', 'Fe2O3t'}
    unit_map = unit_map or {}
    ignore_set = set(ignore_cols or [])
    rename_map = rename_map or {}

    def _custom_target(orig_col):
        """Trimmed custom rename target for *orig_col*, or '' if none."""
        v = rename_map.get(orig_col, '')
        return v.strip() if isinstance(v, str) else ''

    result = raw_df.copy()
    # Drop user-Ignored columns first — they never reach the rename loop, so
    # they can't accidentally claim or shadow an internal name a sibling
    # column (the auto-mapper kept) is supposed to receive.
    if ignore_set:
        result = result.drop(columns=[c for c in ignore_set if c in result.columns],
                             errors='ignore')

    # effective_target_name → (orig_col, scale_factor, conv); first-encountered wins.
    # The dedup key is the *final output column name* (custom rename if set,
    # otherwise the registry's internal_name) — this lets two columns mapping
    # to the same registry target survive when one has a Rename-to override.
    seen: dict = {}
    # display_label → (orig_col, source_unit) for iron resolution below
    _iron_map: dict = {}

    for orig_col, display_label in mapping.items():
        if orig_col in ignore_set:
            continue
        if orig_col not in result.columns or display_label == _DP_KEEP_ORIGINAL:
            continue
        if display_label not in _DP_DISPLAY_TO_INTERNAL:
            continue
        internal_name, target_unit, conv = _DP_DISPLAY_TO_INTERNAL[display_label]
        # Source unit: explicit from the editor's Unit column, or detected
        # from the original header, or the registry default.
        source_unit = unit_map.get(orig_col) or _dp_detect_unit(orig_col) or target_unit
        custom = _custom_target(orig_col)
        # Iron resolver only handles columns destined for the canonical FeO
        # column (i.e. no custom rename). With a custom rename the user wants
        # the values preserved under that name, so route through the regular
        # rename path (unit conversion still applies).
        if display_label in _IRON_LABELS and not custom:
            if display_label not in _iron_map:
                _iron_map[display_label] = (orig_col, source_unit)
            continue
        # Compute scale factor: unit conversion × element-to-oxide factor.
        scale = _dp_unit_scale(source_unit, target_unit)
        if conv and conv[0] == 'elem_oxide':
            scale *= conv[1]
        effective_target = custom if custom else internal_name
        if effective_target not in seen:
            seen[effective_target] = (orig_col, scale, conv)

    for effective_target, (orig_col, scale, conv) in seen.items():
        # Defensive: if a column already exists with the rename target's
        # name (e.g. file has both 'SIO2wt' AND a literal 'SiO2' column,
        # both auto-mapped to internal 'SiO2'), drop the pre-existing one
        # so the rename produces a clean unique column rather than a
        # duplicate-name pandas mess.
        if effective_target != orig_col and effective_target in result.columns:
            result = result.drop(columns=[effective_target])
        if conv is not None and conv[0] == 'utm':
            # UTM → DD conversion happens separately below; just rename.
            result = result.rename(columns={orig_col: effective_target})
            continue
        if scale == 1.0:
            result = result.rename(columns={orig_col: effective_target})
        else:
            result[orig_col] = pd.to_numeric(result[orig_col], errors='coerce') * scale
            result = result.rename(columns={orig_col: effective_target})

    # ── Iron resolution ──────────────────────────────────────────────────────
    # Each iron source column may need its own unit conversion BEFORE the
    # speciation arithmetic. Convert once into wt% values.
    def _num_wt(entry):
        if entry is None:
            return None, None
        col, src_u = entry
        if col not in result.columns:
            return None, None
        vals = pd.to_numeric(result[col], errors='coerce')
        if src_u and src_u != 'wt%':
            vals = vals * _dp_unit_scale(src_u, 'wt%')
        return vals, col

    _feot_vals,   _feot_col   = _num_wt(_iron_map.get('FeOt'))
    _feo_vals,    _feo_col    = _num_wt(_iron_map.get('FeO'))
    _fe2o3_vals,  _fe2o3_col  = _num_wt(_iron_map.get('Fe2O3'))
    _fe2o3t_vals, _fe2o3t_col = _num_wt(_iron_map.get('Fe2O3t'))

    if _feot_vals is not None:
        # FeOt directly mapped — gold standard, use as-is
        feot_resolved = _feot_vals
    elif _fe2o3t_vals is not None:
        # Total iron as Fe2O3 — convert to FeO equivalent
        feot_resolved = _fe2o3t_vals * FE2O3_TO_FEO
    elif _feo_vals is not None and _fe2o3_vals is not None:
        # Speciated pair: FeOt = FeO + Fe2O3 × 0.8998
        feot_resolved = _feo_vals.fillna(0) + _fe2o3_vals.fillna(0) * FE2O3_TO_FEO
        feot_resolved[_feo_vals.isna() & _fe2o3_vals.isna()] = np.nan
    elif _feo_vals is not None:
        feot_resolved = _feo_vals   # FeO only (treat as FeOt)
    elif _fe2o3_vals is not None:
        feot_resolved = _fe2o3_vals * FE2O3_TO_FEO
    else:
        feot_resolved = None

    # Write resolved FeO (= FeOt equivalent) and drop all source iron columns
    _iron_orig_cols = [c for c in [_feot_col, _feo_col, _fe2o3_col, _fe2o3t_col] if c]
    result = result.drop(columns=[c for c in _iron_orig_cols if c in result.columns], errors='ignore')
    # Also drop any Fe2O3/Fe2O3T that the generic loop may have already renamed
    result = result.drop(columns=[c for c in ['Fe2O3', 'Fe2O3T'] if c in result.columns], errors='ignore')
    if feot_resolved is not None:
        result['FeO'] = feot_resolved.values
    if utm_info and 'UTM_Easting' in result.columns and 'UTM_Northing' in result.columns:
        try:
            e = pd.to_numeric(result['UTM_Easting'], errors='coerce').values
            n = pd.to_numeric(result['UTM_Northing'], errors='coerce').values
            lats, lons = _dp_utm_to_dd(e, n, utm_info[0], utm_info[1])
            result['Lat'] = lats; result['Lon'] = lons
            result = result.drop(columns=['UTM_Easting', 'UTM_Northing'], errors='ignore')
        except Exception:
            pass
    result = coerce(result)
    result = _dp_infer_age_from_geologic_time(result)
    # Synthesise Arc_or_Segment from separate Arc / Segment columns (backward compat)
    if 'Arc' in result.columns or 'Segment' in result.columns:
        _arc = result.get('Arc', pd.Series(pd.NA, index=result.index, dtype=object))
        _seg = result.get('Segment', pd.Series(pd.NA, index=result.index, dtype=object))
        result['Arc_or_Segment'] = _arc.combine_first(_seg)
    # Synthesise Rock_Type from Lithology_Type so downstream model classification works
    if 'Rock_Type' not in result.columns and 'Lithology_Type' in result.columns:
        result['Rock_Type'] = result['Lithology_Type']
    # Null BDL values before computing ratios — negatives are unphysical concentrations
    result = _dp_null_below_detection(result)
    # Compute geochemical ratios from element columns
    result = _dp_compute_ratios(result)
    if 'Sample_ID' not in result.columns:
        result.insert(0, 'Sample_ID', [f'Sample_{i+1}' for i in range(len(result))])
    return ensure_unique_columns(result)


_DP_LAT_LON_LABELS = {'Lat', 'Lon', 'Lat min', 'Lat max', 'Lon min', 'Lon max'}

# ── Data-Quality helpers ───────────────────────────────────────────────────────

_DP_MAJOR_OXIDES = ['SiO2','TiO2','Al2O3','FeO','Fe2O3','Fe2O3T','MnO','MgO','CaO','Na2O','K2O','P2O5','LOI']
_DP_OXIDE_TOTAL_MIN = 94.0
_DP_OXIDE_TOTAL_MAX = 106.0
_DP_ELEMENT_BOUNDS: dict = {
    'SiO2':(30,85),'TiO2':(0,10),'Al2O3':(0,30),'FeO':(0,25),'Fe2O3':(0,25),
    'Fe2O3T':(0,25),'MnO':(0,3),'MgO':(0,50),'CaO':(0,30),'Na2O':(0,12),
    'K2O':(0,12),'P2O5':(0,5),'LOI':(-5,25),
    'La':(0,5000),'Ce':(0,10000),'Sr':(0,20000),'Ba':(0,20000),
    'Rb':(0,2000),'Nb':(0,2000),'Ta':(0,200),'Th':(0,500),'U':(0,200),
    'Zr':(0,5000),'Hf':(0,500),'Y':(0,1000),
    'Cr':(0,5000),'Ni':(0,5000),'Co':(0,1000),'Sc':(0,200),'V':(0,2000),
}

# Internal names of columns that represent actual geochemical measurements
# (major oxides, REE, trace elements, crustal-thickness target, age).
# Used to strip ratios, coordinates, metadata, and computed columns from the QA chart.
_DP_QA_MEASUREMENT_COLS: set = {
    iname for _dl, iname, cat, _unit, _conv in _DP_REGISTRY
    if cat in ('major', 'ree', 'trace', 'target', 'age')
}

def _dp_qa_report(df: pd.DataFrame) -> dict:
    """Return a dict of QA metrics for a mapped DataFrame."""
    n = max(len(df), 1)
    # Fill %
    fill_pct = {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce') if c in _DP_ELEMENT_BOUNDS or c in _DP_MAJOR_OXIDES else df[c]
        fill_pct[c] = round(100.0 * s.notna().sum() / n, 1)
    # Oxide total
    ox_cols = [c for c in _DP_MAJOR_OXIDES if c in df.columns]
    oxide_total = None; oxide_flags = pd.Series(dtype=bool)
    if ox_cols:
        num = df[ox_cols].apply(pd.to_numeric, errors='coerce')
        oxide_total = num.sum(axis=1, min_count=1)
        oxide_flags = oxide_total.notna() & ((oxide_total < _DP_OXIDE_TOTAL_MIN) | (oxide_total > _DP_OXIDE_TOTAL_MAX))
    # Element outliers (beyond hard physical bounds)
    outlier_counts: dict = {}
    for col, (lo, hi) in _DP_ELEMENT_BOUNDS.items():
        if col not in df.columns: continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        bad = ((s < lo) | (s > hi)).sum()
        if bad: outlier_counts[col] = int(bad)
    return {
        'n_rows': len(df),
        'fill_pct': fill_pct,
        'oxide_cols': ox_cols,
        'oxide_total': oxide_total,
        'oxide_flags': oxide_flags,
        'n_oxide_bad': int(oxide_flags.sum()) if len(oxide_flags) else 0,
        'outlier_counts': outlier_counts,
    }


def _dp_qa_panel(df: pd.DataFrame, key_prefix: str) -> None:
    """Render a compact data-quality panel inside an expander."""
    qa = _dp_qa_report(df)
    n = qa['n_rows']
    # Only show mapped geochemical measurement columns — exclude ratios, coordinates,
    # metadata, and anything not in the registry.
    cols_present = [c for c in df.columns if c in _DP_QA_MEASUREMENT_COLS]
    # Header metrics
    m1, m2, m3 = st.columns(3)
    m1.metric('Rows', f'{n:,}')
    m2.metric('Oxide total outliers', qa['n_oxide_bad'],
              help=f'Rows where major-oxide sum is outside {_DP_OXIDE_TOTAL_MIN}–{_DP_OXIDE_TOTAL_MAX} wt%')
    m3.metric('Columns with bad values', len(qa['outlier_counts']),
              help='Columns with at least one value outside physical plausibility bounds')

    # Fill % bar chart (only columns present in data)
    fill_items = [(c, qa['fill_pct'].get(c, 0.0)) for c in cols_present if c in qa['fill_pct']]
    if fill_items:
        fill_df = pd.DataFrame(fill_items, columns=['Column', 'Fill %'])
        fill_df['Colour'] = fill_df['Fill %'].apply(
            lambda v: '#22c55e' if v >= 90 else ('#f59e0b' if v >= 50 else '#ef4444'))
        fig_fill = go.Figure(go.Bar(
            x=fill_df['Column'], y=fill_df['Fill %'],
            marker_color=fill_df['Colour'],
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
        ))
        fig_fill.update_layout(
            height=220, margin=dict(l=10, r=10, t=10, b=60),
            yaxis=dict(range=[0, 100], title='Fill %'),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            template='plotly_white',
        )
        st.plotly_chart(fig_fill, use_container_width=True, key=f'{key_prefix}_qa_fill')

    # Oxide total histogram
    if qa['oxide_total'] is not None and qa['oxide_total'].notna().sum() > 0:
        tot = qa['oxide_total'].dropna()
        fig_ox = go.Figure(go.Histogram(
            x=tot, nbinsx=40,
            marker_color='#6366f1', opacity=0.8,
            hovertemplate='Total %{x:.1f} wt%: %{y} samples<extra></extra>',
        ))
        fig_ox.add_vrect(x0=_DP_OXIDE_TOTAL_MIN, x1=_DP_OXIDE_TOTAL_MAX,
                         fillcolor='#bbf7d0', opacity=0.25, line_width=0,
                         annotation_text='accepted range', annotation_position='top left',
                         annotation_font_size=10)
        fig_ox.update_layout(
            height=210, margin=dict(l=10, r=10, t=20, b=40),
            xaxis_title='Oxide total [wt%]', yaxis_title='N samples',
            template='plotly_white',
        )
        st.plotly_chart(fig_ox, use_container_width=True, key=f'{key_prefix}_qa_oxtot')

    # Outlier table
    if qa['outlier_counts']:
        st.markdown('**Values outside physical bounds**')
        out_df = pd.DataFrame(
            [(c, n, f'{_DP_ELEMENT_BOUNDS[c][0]}–{_DP_ELEMENT_BOUNDS[c][1]}')
             for c, n in sorted(qa['outlier_counts'].items(), key=lambda x: -x[1])],
            columns=['Column', 'N flagged', 'Expected range'],
        )
        st.dataframe(out_df, hide_index=True, use_container_width=True)

def _dp_fmt_val(v, max_dp: int = 2) -> str:
    """Format a cell value: smart decimal trimming (max max_dp dp, no trailing zeros)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    try:
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            return str(v)
        if fv == int(fv):
            return str(int(fv))
        best = f'{fv:.{max_dp}f}'.rstrip('0').rstrip('.')
        # If 1 dp is exact enough, use it (only when max_dp >= 2)
        if max_dp >= 2:
            s1 = f'{fv:.1f}'.rstrip('0').rstrip('.')
            if abs(float(s1) - fv) < 0.005:
                return s1
        return best
    except (ValueError, TypeError):
        s = str(v)
        return (s[:20] + '…') if len(s) > 20 else s


def _dp_preview_html(orig_cols, mapping, confirmed_set, raw_df, search_ph='', max_cols=60, n_rows=5):
    """Single scrollable table: 2 header rows (label / original) + n data rows.
    Clicking a canonical header cell sets the filter input identified by search_ph."""
    cols = [c for c in orig_cols[:max_cols] if c in raw_df.columns]
    th_canon = []; th_orig = []
    for c in cols:
        disp = mapping.get(c, _DP_KEEP_ORIGINAL)
        if disp == _DP_KEEP_ORIGINAL:
            fg = '#6b7280'; bg = '#f3f4f6'; label = c
        elif c in confirmed_set:
            fg = '#1d4ed8'; bg = '#eff6ff'; label = disp
        else:
            fg = '#dc2626'; bg = '#fef2f2'; label = disp
        sl = html.escape(str(label)); so = html.escape(str(c))
        # JS onclick: set filter input value + scroll to it
        _jsc = json.dumps(c)   # JSON-escaped column name (handles quotes/specials)
        if search_ph:
            _onclick = (
                f"(function(){{var i=document.querySelector('input[placeholder=&quot;{search_ph}&quot;]');"
                f"if(i){{var s=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;"
                f"s.call(i,{_jsc});i.dispatchEvent(new Event('input',{{bubbles:true}}));"
                f"i.scrollIntoView({{behavior:'smooth',block:'center'}});"
                f"this.style.outline='3px solid #f59e0b';"
                f"setTimeout((function(el){{return function(){{el.style.outline='';}}}})(this),1200);}}}}).call(this);"
            )
            _cursor = 'cursor:pointer;'
            _tip = f'title="click to filter mapping table → {html.escape(c)}"'
        else:
            _onclick = ''; _cursor = ''; _tip = f'title="{sl}"'
        th_canon.append(
            f'<th {_tip} onclick="{_onclick}" style="color:{fg};background:{bg};padding:5px 10px;'
            f'font-size:11px;font-weight:700;border:1px solid #d1d5db;white-space:nowrap;'
            f'position:sticky;top:0;z-index:2;{_cursor}">{sl}</th>')
        th_orig.append(
            f'<th title="{so}" style="background:#e5e7eb;color:#374151;padding:5px 10px;'
            f'font-size:11px;font-weight:500;border:1px solid #d1d5db;white-space:nowrap;'
            f'position:sticky;top:22px;z-index:2">{so}</th>')
    _max_dps = [4 if mapping.get(c, _DP_KEEP_ORIGINAL) in _DP_LAT_LON_LABELS else 2 for c in cols]
    data_rows = []
    for row in raw_df.head(n_rows).itertuples(index=False, name=None):
        row_dict = dict(zip(raw_df.columns, row))
        cells = []
        for c, _max_dp in zip(cols, _max_dps):
            fmted = _dp_fmt_val(row_dict.get(c, ''), _max_dp)
            safe  = html.escape(fmted)
            cells.append(
                f'<td title="{safe}" style="padding:3px 10px;border:1px solid #d1d5db;'
                f'font-size:11px;color:#374151;white-space:nowrap">{safe}</td>')
        data_rows.append('<tr>' + ''.join(cells) + '</tr>')
    return (
        '<div style="overflow-x:auto;margin:4px 0 12px 0">'
        '<table style="border-collapse:collapse;font-family:monospace">'
        '<thead>'
        '<tr>' + ''.join(th_canon) + '</tr>'
        '<tr>' + ''.join(th_orig)  + '</tr>'
        '</thead>'
        '<tbody>' + ''.join(data_rows) + '</tbody>'
        '</table></div>'
    )

def read_crust_grid(file_or_path):
    name = getattr(file_or_path,'name',str(file_or_path)).lower()
    df = pd.read_excel(file_or_path) if name.endswith(('.xlsx','.xls')) else pd.read_csv(file_or_path)
    df = std_crust_cols(df)
    df = coerce(df)
    for c in ['CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Vp','CRUST1_Density']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return ensure_unique_columns(df)

@st.cache_data(show_spinner=False)
def read_default_crust_grid(path='CRUST_1_0_excel.csv'):
    p = Path(__file__).resolve().parent / path
    if not p.exists():
        p = Path(path)
    return read_crust_grid(p) if p.exists() else pd.DataFrame()

# ─── LithoRef18 (Alfonso 2019) reference grid ────────────────────────────────

def _parse_lithoref18_text(text):
    col_names = None; rows = []
    for line in text.replace('\r','').split('\n'):
        s = line.strip()
        if not s: continue
        if col_names is None and 'LONG' in s and 'MOHO' in s:
            col_names = s.split(); continue
        if col_names is None: continue
        try: vals = [float(v) for v in s.split()]
        except ValueError: continue
        if len(vals) >= 4: rows.append(vals[:len(col_names)])
    if not rows: return pd.DataFrame()
    cols = col_names or ['LONG','LAT','ELEVATION','MOHO','LAB','RHO_C','RHO_L','RHO_SL','BOTTOM','GEOID','FA','G_zz','G_xx','G_yy']
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])]).rename(columns={'LONG':'Lon','LAT':'Lat'})
    # Crustal thickness = vertical distance from surface to Moho (ELEVATION and MOHO both metres relative to sea level)
    df['LithoRef18_Total_Crust_km'] = (df['ELEVATION'] - df['MOHO']) / 1000.0
    if 'LAB' in df:
        df['LithoRef18_LAB_km'] = (df['ELEVATION'] - df['LAB']) / 1000.0
    keep = ['Lon','Lat','LithoRef18_Total_Crust_km'] + [c for c in ['LithoRef18_LAB_km'] if c in df]
    return df[keep].dropna(subset=['Lat','Lon','LithoRef18_Total_Crust_km'])

def _load_lithoref18_from(source):
    import zipfile, io
    if hasattr(source, 'read'):
        raw = source.read()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                xyz = next((n for n in z.namelist() if n.endswith('.xyz')), None)
                if xyz: return _parse_lithoref18_text(z.read(xyz).decode('utf-8','replace'))
        except Exception: pass
        return _parse_lithoref18_text(raw.decode('utf-8','replace'))
    p = Path(source)
    if p.suffix.lower() == '.zip':
        with zipfile.ZipFile(p) as z:
            xyz = next((n for n in z.namelist() if n.endswith('.xyz')), None)
            if not xyz: return pd.DataFrame()
            return _parse_lithoref18_text(z.read(xyz).decode('utf-8','replace'))
    with open(p,'r',encoding='utf-8',errors='replace') as f:
        return _parse_lithoref18_text(f.read())

@st.cache_data(show_spinner=False)
def read_default_lithoref18():
    for p in [DEFAULT_LITHOREF18_XYZ, DEFAULT_LITHOREF18_ZIP, _ONEDRIVE_LITHOREF18_ZIP]:
        if p.exists():
            try:
                df = _load_lithoref18_from(p)
                if not df.empty: return df
            except Exception: pass
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def _triangulated_interpolator(grid_id, lats, lons, vals):
    if _LinearNDInterp is None: return None
    pts = np.column_stack([lats, lons])
    ok = np.isfinite(pts).all(axis=1) & np.isfinite(vals)
    if ok.sum() < 3: return None
    return _LinearNDInterp(pts[ok], vals[ok], fill_value=np.nan)

@st.cache_resource(show_spinner=False)
def _fallback_kdtree(grid_id, lats, lons, vals):
    """Nearest-cell KDTree for points outside the triangulation convex hull. Cached per grid."""
    if cKDTree is None: return None, None
    valid = np.isfinite(vals)
    if not valid.any(): return None, None
    return cKDTree(np.column_stack([lats[valid], lons[valid]])), vals[valid]

def grid_interpolate(sample_lats, sample_lons, grid_df, value_col, grid_id='grid'):
    """Triangulated linear interpolation at sample points. Falls back to nearest-cell outside the convex hull."""
    glat = grid_df['Lat'].to_numpy(dtype=float)
    glon = grid_df['Lon'].to_numpy(dtype=float)
    gval = pd.to_numeric(grid_df[value_col], errors='coerce').to_numpy(dtype=float)
    interp = _triangulated_interpolator(grid_id, glat, glon, gval)
    pts = np.column_stack([np.asarray(sample_lats, float), np.asarray(sample_lons, float)])
    result = np.asarray(interp(pts), dtype=float) if interp is not None else np.full(len(pts), np.nan)
    nan_mask = ~np.isfinite(result)
    if nan_mask.any():
        tree, valid_vals = _fallback_kdtree(grid_id, glat, glon, gval)
        if tree is not None:
            _, idxs = tree.query(pts[nan_mask])
            result[nan_mask] = valid_vals[idxs]
    return result

@st.cache_data(show_spinner=False)
def attach_grid_as_target(df, grid_df, value_col, target_name, grid_id='grid'):
    """Attach any reference grid value as a training target using triangulated linear interpolation."""
    if df.empty or grid_df.empty or value_col not in grid_df or not {'Lat','Lon'}.issubset(df):
        return df, None, 'Needs Lat, Lon and a valid reference grid.'
    out = df.copy().reset_index(drop=True)
    coords = out[['Lat','Lon']].apply(pd.to_numeric, errors='coerce')
    ok = coords['Lat'].notna() & coords['Lon'].notna()
    if not ok.any():
        return out, None, 'Needs at least one row with numeric Lat and Lon.'
    out[target_name] = np.nan
    out.loc[ok, target_name] = grid_interpolate(
        coords.loc[ok,'Lat'].to_numpy(), coords.loc[ok,'Lon'].to_numpy(),
        grid_df, value_col, grid_id,
    )
    return out, target_name, None

# ─────────────────────────────────────────────────────────────────────────────

def _app_dir():
    """Directory containing Mohometer.py — reliable regardless of launch directory."""
    return Path(__file__).resolve().parent

def _find_file(names):
    """Return the first matching file, checking beside the script before cwd."""
    app_dir = _app_dir()
    for name in names:
        p = app_dir / name
        if p.exists():
            return p
        p = Path(name)
        if p.exists():
            return p
    return None

def find_training():
    return _find_file(['GuoYang_2023_Model.xlsx','Table S1(1).xlsx','Table S1.xlsx'])

def find_zou_training():
    return _find_file(['Zou_2021_Model.xlsx','Zou_2021_Model.csv','Zou2021_Model.xlsx','Zou2021_Model.csv','Zou_2021.xlsx','Zou_2021.csv'])

def find_luffi_training():
    # Prefer the full T1 primary dataset (~35k samples) over the 121-row arc averages.
    # Arc averages are kept for GAME calibration only (load_game_calibration).
    return _find_file([
        'LuffiDucea_2022_T1.xlsx',
        'Luffi & Ducea - Supplementary Table T1 (global arc primary datase.xlsx',
        'LuffiDucea_2022_T1.csv',
        'LuffiDucea_2022_Calibration.csv',
        'luffi_ducea_2022_game_calibration.csv',
        'LuffiDucea_2022_Model.csv',
        'LuffiDucea_2022_Model.xlsx',
    ])


# ── Built-in dataset display name → filename stem ────────────────────────────
# When a user loads one of these via the Prepare-tab "Load reference dataset"
# button, the prepared dataframe lands in the pool under its stem (e.g.
# 'GuoYang_2023_Model'). The Model/Validate dropdowns continue to show only
# the friendly built-in name (e.g. 'Guo & Yang (2023)'); read_training_source
# checks for a prepared version under the matching stem and uses it
# preferentially over the raw on-disk file. This way the user sees ONE entry
# per dataset, with their column-mapping decisions applied automatically.
_BUILTIN_TO_FILE_STEM = {
    'Guo & Yang (2023)':  'GuoYang_2023_Model',
    'Zou et al. (2021)':  'Zou_2021_Model',
    'Luffi & Ducea (2022)': 'LuffiDucea_2022_T1',
}

def _load_luffi_training(path):
    """Load a Luffi & Ducea (2022) file as a training dataframe.

    Handles two formats:
      - Full T1 primary dataset (~35k rows): has 'CRUST1 Moho depth (km)',
        'Latitude', 'Longitude', 'GMRT elevation (km)' plus cleaned element
        columns in '(wt%)' / '(ppm)' style in the rightmost block.
      - 121-row arc averages (GAME calibration table): has 'elevation (km)'
        and pre-computed ratios; target derived from H = 6.79×elev + 26.40.
    """
    # Read the sheet named 'Global arc dataset' if present (T1 xlsx), else plain read
    suffix = Path(path).suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        try:
            xl = pd.ExcelFile(path)
            sheet = 'Global arc dataset' if 'Global arc dataset' in xl.sheet_names else xl.sheet_names[0]
            raw = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    # ── T1 primary dataset path ───────────────────────────────────────────────
    if 'CRUST1 Moho depth (km)' in raw.columns:
        # Spatial + target columns
        rename = {
            'Latitude': 'Lat', 'Longitude': 'Lon',
            'GMRT elevation (km)': 'Elevation_km',
            'CRUST1 Moho depth (km)': 'Crust_Thickness',
        }
        # Raw ppm/wt columns → canonical element names. The Luffi T1
        # primary dataset ships ~70 trace-element columns and ~20
        # radiogenic / He isotope ratios in addition to the ~10 major
        # oxides exposed in the cleaned '(wt%)' / '(ppm)' block at the
        # right of the sheet. The app previously only mapped REE +
        # ~20 trace elements; everything else (PGE, halogens, metalloids,
        # most transition metals, isotope ratios, volatiles) was being
        # silently dropped on load. Extending the mapping pulls them all
        # through under canonical column names so models, group filters,
        # and the Custom XY graph can use them.
        _T1_ELEMENT_MAP = {
            # ── REE (rare-earth elements) ────────────────────────────────
            'LAppm':'La','CEppm':'Ce','PRppm':'Pr','NDppm':'Nd','SMppm':'Sm',
            'EUppm':'Eu','GDppm':'Gd','TBppm':'Tb','DYppm':'Dy','HOppm':'Ho',
            'ERppm':'Er','TMppm':'Tm','YBppm':'Yb','LUppm':'Lu',
            # ── LILE / HFSE / actinides ──────────────────────────────────
            'SRppm':'Sr','Yppm':'Y','RBppm':'Rb','BAppm':'Ba',
            'NBppm':'Nb','HFppm':'Hf','TAppm':'Ta','THppm':'Th','Uppm':'U',
            'ZRppm':'Zr',
            # ── First-row transition metals ──────────────────────────────
            'SCppm':'Sc','Vppm':'V','CRppm':'Cr','COppm':'Co',
            'NIppm':'Ni','CUppm':'Cu','ZNppm':'Zn','GAppm':'Ga','PBppm':'Pb',
            'TIppm':'Ti_ppm','MNppm':'Mn_ppm','FEppm':'Fe_ppm',
            # ── Light & post-transition (incl. metalloids) ───────────────
            'LIppm':'Li','BEppm':'Be','CSppm':'Cs','Bppm':'B',
            'GEppm':'Ge','ASppm':'As','SEppm':'Se',
            # ── Halogens / volatiles (ppm-level) ─────────────────────────
            'BRppm':'Br','Ippm':'I',
            # ── Light major-element ppm forms (kept under _ppm so the
            # canonical wt% columns are unaffected) ──────────────────────
            'NAppm':'Na_ppm','MGppm':'Mg_ppm','ALppm':'Al_ppm','Pppm':'P_ppm',
            'Sppm':'S_ppm','CLppm':'Cl_ppm','Kppm':'K_ppm','CAppm':'Ca_ppm',
            # ── Refractory metals + PGE + late transition ────────────────
            'MOppm':'Mo','RUppm':'Ru','RHppm':'Rh','PDppm':'Pd','AGppm':'Ag',
            'CDppm':'Cd','INppm':'In','SNppm':'Sn','SBppm':'Sb','TEppm':'Te',
            'Wppm':'W','REppm':'Re','OSppm':'Os','IRppm':'Ir','PTppm':'Pt',
            'AUppm':'Au','HGppm':'Hg','TLppm':'Tl','BIppm':'Bi',
            # ── Carbon (ppm) — distinct from CO2 wt% ─────────────────────
            'Cppm':'C_ppm','Fppm':'F_ppm','CO2ppm':'CO2_ppm',
        }
        for src, dst in _T1_ELEMENT_MAP.items():
            # Only rename if the clean '(ppm)' column doesn't already exist
            clean = f'{dst} (ppm)'
            if src in raw.columns and clean not in raw.columns and dst not in raw.columns:
                rename[src] = dst

        # ── Radiogenic + noble-gas isotope ratios ────────────────────────
        # Geochemists use these as tracers of mantle source / crustal
        # contamination — they're independent dimensions a model can
        # use as features. Mapped to spelled-out canonical names so
        # autocomplete and Custom-XY pickers stay readable.
        _T1_ISOTOPE_MAP = {
            'ND143_ND144':       'Nd143_Nd144',
            'ND143_ND144_INI':   'Nd143_Nd144_initial',
            'EPSILON_ND':        'epsilon_Nd',
            'SR87_SR86':         'Sr87_Sr86',
            'SR87_SR86_INI':     'Sr87_Sr86_initial',
            'PB206_PB204':       'Pb206_Pb204',
            'PB206_PB204_INI':   'Pb206_Pb204_initial',
            'PB207_PB204':       'Pb207_Pb204',
            'PB207_PB204_INI':   'Pb207_Pb204_initial',
            'PB208_PB204':       'Pb208_Pb204',
            'PB208_PB204_INI':   'Pb208_Pb204_initial',
            'HF176_HF177':       'Hf176_Hf177',
            'OS184_OS188':       'Os184_Os188',
            'OS186_OS188':       'Os186_Os188',
            'OS187_OS186':       'Os187_Os186',
            'OS187_OS188':       'Os187_Os188',
            'RE187_OS186':       'Re187_Os186',
            'RE187_OS188':       'Re187_Os188',
            'HE3_HE4':           'He3_He4',
            'HE4_HE3':           'He4_He3',
            'HE3_HE4_R_Ra':      'He_R_over_Ra',
            'HE4_HE3_R_Ra':      'He_invR_over_Ra',
            'K40_AR40':          'K40_Ar40',
            'AR40_K40':          'Ar40_K40',
        }
        for src, dst in _T1_ISOTOPE_MAP.items():
            if src in raw.columns and dst not in raw.columns:
                rename[src] = dst

        # ── Volatiles + LOI (wt%) ────────────────────────────────────────
        # H2O total, LOI, CO2 are commonly treated as model features and
        # appear in proxy formulas (e.g. Farner & Lee). Bring them through
        # under canonical names so the Prepare-tab mapping doesn't have
        # to special-case them per upload.
        _T1_VOLATILE_MAP = {
            'H2OTwt':       'H2O_total',
            'H2OPwt':       'H2O_plus',
            'H2OMwt':       'H2O_minus',
            'H2Owt':        'H2O',
            'CO2wt':        'CO2',
            'CO1wt':        'CO',
            'LOIwt':        'LOI',
            'VOLATILESwt':  'Volatiles_total',
            'Owt':          'O',
            'OTHERSwt':     'Others',
            'Fwt':          'F',
            'CLwt':         'Cl',
            'CL2wt':        'Cl2',
            'OHwt':         'OH',
            'CH4wt':        'CH4',
            'SO2wt':        'SO2',
            'SO3wt':        'SO3',
            'SO4wt':        'SO4',
            'Swt':          'S',
            'B2O3wt':       'B2O3',
            'CR2O3wt':      'Cr2O3',
            'NIOwt':        'NiO',
            'FE2O3wt':      'Fe2O3',
            'FEOTwt':       'FeOt',
        }
        for src, dst in _T1_VOLATILE_MAP.items():
            if src in raw.columns and dst not in raw.columns:
                rename[src] = dst
        raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
        # Apply read_table's post-read normalisation directly to the in-memory df.
        df = std_cols(raw)
        if 'Sample_ID' not in df:
            df.insert(0, 'Sample_ID', [f'Sample_{i+1}' for i in range(len(df))])
        df = coerce(df)
        df = iron_to_feo(df)
        for col in ('Crust_Thickness', 'Lat', 'Lon', 'Elevation_km'):
            if col not in df and col in raw.columns:
                df[col] = pd.to_numeric(raw[col], errors='coerce')
        df = df.dropna(subset=['Crust_Thickness']).reset_index(drop=True)
        return df

    # ── 121-row arc-averages path ─────────────────────────────────────────────
    df = std_cols(raw)
    if 'Sample_ID' not in df:
        df.insert(0, 'Sample_ID', [f'Sample_{i+1}' for i in range(len(df))])
    df = coerce(df)
    df = iron_to_feo(df)
    if 'Crust_Thickness' not in df and 'Elevation_km' in df:
        elev = pd.to_numeric(df['Elevation_km'], errors='coerce')
        df['Crust_Thickness'] = (GAME_ALPHA_DEFAULT * elev + GAME_BETA_DEFAULT).round(2)
    if 'Crust_Thickness' in df:
        df = df.dropna(subset=['Crust_Thickness']).reset_index(drop=True)
    return df

def target_col(df):
    for c in ['Crust_Thickness','Crustal thickness','Crustal Thickness']:
        if c in df: return c
    for c in df.columns:
        s = str(c).lower()
        if 'crust1' in s or 'crust_1' in s or 'crust 1' in s:
            continue
        if 'crust' in s and 'thick' in s: return c
    return None

def is_crust1_column(c):
    s = str(c).lower()
    return 'crust1' in s or 'crust_1' in s or 'crust 1' in s or str(c).startswith('Nearest_CRUST1')

def numeric_columns(df):
    return list(dict.fromkeys([c for c in df.columns if has_numeric_column(df, c)]))

def attach_crust1_target(df, target_name='CRUST1_Target_km'):
    if df.empty or not {'Lat','Lon'}.issubset(df):
        return df, None, 'CRUST1.0 target needs Lat and Lon columns.'
    grid = read_default_crust_grid(str(DEFAULT_CRUST1_GRID))
    if grid.empty:
        return df, None, f'CRUST1.0 target needs {DEFAULT_CRUST1_GRID} beside the app.'
    out = df.copy().reset_index(drop=True)
    out[target_name] = np.nan
    coords = out[['Lat','Lon']].apply(pd.to_numeric, errors='coerce')
    ok = coords['Lat'].notna() & coords['Lon'].notna()
    if not ok.any():
        return out, None, 'CRUST1.0 target needs at least one row with numeric Lat and Lon.'
    samples = out.loc[ok].copy()
    samples['Lat'] = coords.loc[ok, 'Lat']
    samples['Lon'] = coords.loc[ok, 'Lon']
    matched = nearest_crust_matches(samples, grid)
    out.loc[ok, target_name] = matched['CRUST1_Benchmark_km'].values
    return out, target_name, None

def target_column_controls(prefix, df):
    detected = target_col(df)
    nums = numeric_columns(df)
    options = []
    if detected:
        options.append(detected)
    has_latlon = {'Lat','Lon'}.issubset(df)
    crust1_option  = '__CRUST1_INTERP__'
    lithoref_option = '__LITHOREF18_INTERP__'
    if has_latlon and DEFAULT_CRUST1_GRID.exists():
        options.append(crust1_option)
    lithoref_grid = read_default_lithoref18()
    if has_latlon and not lithoref_grid.empty:
        options.append(lithoref_option)
    options += [c for c in nums if c != detected and not is_crust1_column(c)]
    options = list(dict.fromkeys(options))
    if not options:
        st.warning(f'{prefix} needs a numeric crustal-thickness target column, or Lat/Lon so a reference grid can be used as the target.')
        return df, None
    def target_label(o):
        if o == crust1_option:   return 'CRUST1.0 — triangulated linear interpolation (1°×1° grid)'
        if o == lithoref_option: return 'LithoRef18 / Alfonso 2019 — triangulated linear interpolation (2°×2° grid)'
        if o == detected:        return f'Detected target: {o}'
        return f'Column: {o}'
    st.markdown(f'**{prefix} training target**')
    left, mid, right = st.columns([0.7,2.2,0.7])
    with mid:
        choice = st.selectbox('Crustal-thickness target', options, index=0, key=f'{prefix}_target_col', format_func=target_label, label_visibility='collapsed')
    if choice == crust1_option:
        grid = read_default_crust_grid(str(DEFAULT_CRUST1_GRID))
        out, target, err = attach_grid_as_target(df, grid, 'CRUST1_Total_Crust_km', 'CRUST1_Target_km', 'CRUST1.0')
        if err: st.warning(err); return df, None
        mid.caption('CRUST1.0 total crustal thickness — triangulated linear interpolation from the 1°×1° grid.')
        return out, target
    if choice == lithoref_option:
        out, target, err = attach_grid_as_target(df, lithoref_grid, 'LithoRef18_Total_Crust_km', 'LithoRef18_Target_km', 'LithoRef18')
        if err: st.warning(err); return df, None
        mid.caption('LithoRef18 (Alfonso 2019) crustal thickness — triangulated linear interpolation from the 2°×2° grid.')
        return out, target
    return df, choice

def make_regressor(algorithm, seed=42):
    algorithm = str(algorithm).replace('ExtraTrees (Guo-style)','ExtraTrees')
    if algorithm == 'RandomForest':
        return RandomForestRegressor(n_estimators=500,max_features=1.0,random_state=seed,n_jobs=-1)
    if algorithm == 'GradientBoosting':
        return GradientBoostingRegressor(n_estimators=500,random_state=seed)
    if algorithm == 'HistGradientBoosting':
        return HistGradientBoostingRegressor(max_iter=500,random_state=seed)
    if algorithm == 'XGBoost' and XGBRegressor is not None:
        return XGBRegressor(n_estimators=500,learning_rate=0.04,max_depth=5,subsample=0.9,colsample_bytree=0.9,random_state=seed,n_jobs=-1,objective='reg:squarederror')
    return ExtraTreesRegressor(n_estimators=500,max_features=1.0,random_state=seed,n_jobs=-1)

def display_algorithm_label(value):
    if isinstance(value, list):
        return [display_algorithm_label(v) for v in value]
    if isinstance(value, tuple):
        return tuple(display_algorithm_label(v) for v in value)
    if value is None:
        return value
    return str(value).replace('ExtraTrees (Guo-style)','ExtraTrees')

def normalize_widget_state(key_name, options=None):
    if key_name not in st.session_state:
        return
    value = display_algorithm_label(st.session_state[key_name])
    if options is None:
        st.session_state[key_name] = value
        return
    if isinstance(value, list):
        cleaned = [v for v in value if v in options]
        if cleaned:
            st.session_state[key_name] = cleaned
        else:
            del st.session_state[key_name]
        return
    if value in options:
        st.session_state[key_name] = value
    else:
        del st.session_state[key_name]

def resolve_option(value, options, fallback_index=0):
    if not options:
        return None
    value = display_algorithm_label(value)
    if value in options:
        return value
    fallback_index = min(max(int(fallback_index), 0), len(options)-1)
    return options[fallback_index]

@st.cache_resource(show_spinner=False)
def train_model(df, target, features, seed=42, algorithm='ExtraTrees'):
    algorithm = display_algorithm_label(algorithm)
    if target is None:
        raise ValueError('Could not find a crustal thickness target column in the training table.')
    missing = [c for c in features if c not in df]
    if missing:
        raise ValueError(f'Training table is missing columns for this feature set: {", ".join(missing)}')
    clean = df[features+[target]].apply(pd.to_numeric,errors='coerce').dropna()
    if clean.empty:
        raise ValueError('No complete training rows remain after numeric coercion for this feature set.')
    model = Pipeline([('scaler',StandardScaler()),('model',make_regressor(algorithm,seed))])
    model.fit(clean[features], clean[target].values.ravel())
    return model, clean

@st.cache_data(show_spinner=False)
def cv(clean,target,features,seed=42,algorithm='ExtraTrees'):
    algorithm = display_algorithm_label(algorithm)
    if len(clean) < 2:
        return clean[target].values.ravel(), clean[target].values.ravel(), np.nan, np.nan
    X = clean[features].values; y = clean[target].values.ravel(); pred = np.zeros_like(y,dtype=float)
    kf = KFold(n_splits=min(10,len(clean)),shuffle=True,random_state=seed)
    for tr,te in kf.split(X):
        m = Pipeline([('scaler',StandardScaler()),('model',make_regressor(algorithm,seed))])
        m.fit(X[tr],y[tr]); pred[te]=m.predict(X[te])
    return y,pred,r2_score(y,pred),mean_squared_error(y,pred)**0.5

def complete(df, features):
    miss = [c for c in features if c not in df]
    if miss: return pd.Series(False,index=df.index), miss
    return df[features].apply(pd.to_numeric,errors='coerce').notna().all(axis=1), []

def predict(model,df,features,feature_set_name):
    out = df.copy(); ok,miss = complete(out,features); out['H_Guo_ERT_km'] = np.nan
    out['Guo_Feature_Set'] = feature_set_name
    out['Guo_ERT_Status'] = np.where(ok,'Predicted',f'Missing values for {feature_set_name}')
    if ok.any(): out.loc[ok,'H_Guo_ERT_km'] = model.predict(out.loc[ok,features])
    return out

def predict_model_set(models, df):
    out = df.copy()
    first = True
    for name, bundle in models.items():
        features = bundle['features']
        col = f'H_Guo_{name}_km'
        ok,_ = complete(out,features)
        out[col] = np.nan
        if ok.any():
            out.loc[ok,col] = bundle['model'].predict(out.loc[ok,features])
        if first:
            out['H_Guo_ERT_km'] = out[col]
            out['Guo_Feature_Set'] = name
            out['Guo_ERT_Status'] = np.where(ok,'Predicted',f'Missing values for {name}')
            first = False
    return out

def feature_importance(model, features, df=None, target=None):
    """Tree-model feature importance, optionally signed by direction.

    Tree models (RF / ET / GBM / XGB) produce magnitude-only importances —
    they tell you a feature *matters* but not whether **high** or **low**
    values push the prediction up. For a moho-thickness ML model that's
    a real loss of interpretability: it's the difference between "Rb is
    important" and "**high** Rb predicts thicker crust", or "**low** Cr
    predicts thicker crust".

    To recover sign cheaply (no SHAP dependency) we compute Spearman
    rank-correlation between each feature and the target across the
    training rows:
      • ρ > 0  → "higher feature value → higher prediction"  (Direction = +1)
      • ρ < 0  → "higher feature value → lower  prediction"  (Direction = -1)
      • ρ ≈ 0 / non-monotonic → Direction = 0 (sign undefined)

    Spearman handles non-linear monotonic relationships; for genuinely
    non-monotonic features the sign is ambiguous and we fall back to 0
    (rendered as a thin grey bar centred on zero in the chart).

    ``df`` and ``target`` are optional — passing them in adds the
    ``Direction`` and ``Signed_Importance`` columns. Without them the
    output keeps the legacy two-column shape, so older call sites stay
    backward-compatible.
    """
    estimator = model.named_steps.get('model') if hasattr(model,'named_steps') else model
    if estimator is None or not hasattr(estimator,'feature_importances_'):
        return pd.DataFrame()
    imp = pd.DataFrame({'Feature':features,'Importance':estimator.feature_importances_})
    imp['Relative_Importance'] = imp['Importance'] / imp['Importance'].sum()

    # Optional: derive direction from monotonic feature↔target relationship
    if df is not None and target is not None and target in getattr(df, 'columns', []):
        directions = []
        # Coerce target once; reuse mask per feature
        _y = pd.to_numeric(df[target], errors='coerce')
        for feat in features:
            if feat not in df.columns:
                directions.append(0.0)
                continue
            _x = pd.to_numeric(df[feat], errors='coerce')
            _ok = _x.notna() & _y.notna()
            if int(_ok.sum()) < 8:
                directions.append(0.0)
                continue
            try:
                # Spearman rank-corr — robust to non-linear monotonic links
                rho = float(_x[_ok].corr(_y[_ok], method='spearman'))
            except Exception:
                rho = float('nan')
            if not np.isfinite(rho) or abs(rho) < 0.05:
                directions.append(0.0)   # ambiguous / weak / non-monotonic
            else:
                directions.append(1.0 if rho > 0 else -1.0)
        imp['Direction']          = directions
        imp['Signed_Importance']  = imp['Direction'] * imp['Relative_Importance']

    return imp.sort_values('Importance',ascending=False).reset_index(drop=True)

def threshold_feature_set(full_importance, min_ri_pct):
    if full_importance.empty:
        return GUO_FEATURES
    keep = full_importance.loc[full_importance['Relative_Importance'] * 100 >= min_ri_pct, 'Feature'].tolist()
    return keep if keep else [full_importance.iloc[0]['Feature']]

def add_best_fit_lines(fig, df, x_col, y_col, group_col=None):
    groups = [(None, df)] if not group_col or group_col not in df else df.groupby(group_col)
    for name, g in groups:
        g = g[[x_col,y_col]].apply(pd.to_numeric,errors='coerce').dropna()
        if len(g) < 2:
            continue
        slope, intercept = np.polyfit(g[x_col], g[y_col], 1)
        xs = np.linspace(g[x_col].min(), g[x_col].max(), 100)
        label = 'Best fit' if name is None else f'Best fit: {name}'
        fig.add_trace(go.Scatter(x=xs,y=slope*xs+intercept,mode='lines',name=label,line=dict(dash='dot')))
    return fig

def train_configured_models(configs, seed=42):
    models = {}
    validation = []
    importance = []
    for cfg in configs:
        label = display_algorithm_label(cfg['label'])
        algorithm = display_algorithm_label(cfg['algorithm'])
        model, clean = train_model(cfg['df'],cfg['target'],cfg['features'],seed,algorithm)
        models[label] = {'model':model,'clean':clean,'features':cfg['features'],'feature_set':cfg['feature_set'],'algorithm':algorithm}
        validation.append(cv_frame(clean,cfg['target'],cfg['features'],label,seed,algorithm))
        # Pass the cleaned training frame + target so feature_importance
        # can sign each feature by its monotonic relationship to the
        # target (Spearman ρ → Direction = ±1 / 0).
        fi = feature_importance(model,cfg['features'], df=clean, target=cfg['target'])
        if not fi.empty:
            fi.insert(0,'Model',label)
            fi.insert(1,'Feature_Set',cfg['feature_set'])
            fi.insert(2,'Algorithm',algorithm)
            importance.append(fi)
    val_df = pd.concat(validation,ignore_index=True) if validation else pd.DataFrame()
    imp_df = pd.concat(importance,ignore_index=True) if importance else pd.DataFrame()
    return models, val_df, imp_df

# ── Prepped-datasets pool helpers ────────────────────────────────────────────
# The pool is a flat dict {name: DataFrame} stored under
# st.session_state['prepped_datasets']. Prepare adds one entry per uploaded
# file, keyed by the file's stem (e.g. 'My_arc_data'). Role assignment lives
# separately in st.session_state['_dp_dataset_roles']: {stem: {'Model
# (training)', 'Validate', 'Predict'}}.  Tabs filter the pool by role via
# get_pool_for_tab(). The Group tab adds grouped derivatives with auto-named
# keys.
#
# Legacy (removed): role-aggregated bucket entries 'Training data',
# 'Validation data', 'Prediction data'. Listed below only so existing sessions
# with those keys get cleaned up on the next Prepare-tab run.
_PREPARE_POOL_KEYS = ('Training data', 'Validation data', 'Prediction data')

def get_prepped_pool() -> dict:
    """Return the live prepped-datasets pool (mutating the returned dict
    affects session state — by design, so callers can add/remove entries)."""
    return st.session_state.setdefault('prepped_datasets', {})

def list_prepped_dataset_names(only_kind: str | None = None) -> list[str]:
    """Names of all entries in the pool. ``only_kind='prepare'`` returns only
    the canonical Prepare-managed entries; ``'group'`` only the derivatives;
    ``None`` returns everything in insertion order."""
    pool = get_prepped_pool()
    if only_kind == 'prepare':
        return [k for k in pool if k in _PREPARE_POOL_KEYS]
    if only_kind == 'group':
        return [k for k in pool if k not in _PREPARE_POOL_KEYS]
    return list(pool.keys())


def get_pool_for_tab(tab_role: str) -> dict:
    """Return pool entries whose Prepare-tab role assignment includes *tab_role*.

    tab_role should be one of: 'Model (training)', 'Validate', 'Predict'.

    Three categories of pool entries are filtered as follows:

    * **Sheet entries** (individual files uploaded via Prepare, keyed by
      filename stem) → included only when the role assignment for that file
      matches *tab_role*.
    * **Group-derived entries** (e.g. ``MyData [grouped]``) → always
      included; grouping is dataset-level, role-agnostic.
    * **Legacy bucket entries** (``'Training data'`` / ``'Validation data'``
      / ``'Prediction data'``) → always excluded.  These were the old
      role-aggregated concatenations and should never appear in dropdowns
      under the new architecture, even if a stale entry persists from a
      previous session.
    """
    pool = get_prepped_pool()
    roles = st.session_state.get('_dp_dataset_roles', {})
    sheet_keys = set(st.session_state.get('_dp_sheet_pool_keys', []))
    result = {}
    for name, df in pool.items():
        if name in _PREPARE_POOL_KEYS:
            # Stale legacy bucket entry — never expose to tabs.
            continue
        if name in sheet_keys:
            # Individual file uploaded via Prepare — gate on role assignment
            if tab_role in roles.get(name, set()):
                result[name] = df
        else:
            # Group-derived or other non-Prepare entry — always visible.
            result[name] = df
    return result


# ── Reference-dataset helpers (cross-tab) ───────────────────────────────────
# Built-in references and user-registered custom refs are first-class
# datasets across every tab — Group, Model, Validate, Predict. They don't
# need to flow through Prepare unless the user explicitly wants to edit the
# column mapping. The two helpers below give all tabs a uniform way to
# enumerate available datasets and to load whichever one was picked.

_BUILTIN_REF_LABELS = ('Guo & Yang (2023)', 'Zou et al. (2021)', 'Luffi & Ducea (2022)')

def available_reference_labels() -> list:
    """Return display labels for every reference dataset that can be loaded
    right now — built-ins whose disk file is found beside the app, plus
    user-registered custom refs whose path still exists. Order: built-ins
    first (newest convention name), then custom refs."""
    labels = []
    if find_training():        labels.append('Guo & Yang (2023)')
    if find_zou_training():    labels.append('Zou et al. (2021)')
    if find_luffi_training():  labels.append('Luffi & Ducea (2022)')
    try:
        for cr in _dp_load_custom_refs():
            lbl = cr.get('label', '')
            path = cr.get('path', '')
            if lbl and lbl not in labels and path and Path(path).exists():
                labels.append(lbl)
    except Exception:
        pass
    return labels


# Cache the on-disk read for each built-in reference.  read_training_source
# itself already wraps the underlying read_table with @st.cache_data, but it
# also does extra work (Luffi specialised loader, fallback paths) — caching
# the final dataframe by name avoids re-running that on every rerun.
@st.cache_data(show_spinner=False, ttl=3600)
def _load_builtin_ref_cached(name: str) -> 'pd.DataFrame':
    df, _label = read_training_source(name)
    return df if df is not None else pd.DataFrame()

# Cache custom-ref reads keyed by (path, mtime). The mtime is part of the
# cache key so editing the file on disk invalidates the cache automatically.
@st.cache_data(show_spinner=False, ttl=3600)
def _load_custom_ref_cached(path_str: str, mtime: float) -> 'pd.DataFrame':
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    return read_table(p, guo_no_header=False)


def load_dataset_by_name(name: str) -> 'pd.DataFrame':
    """Resolve a dataset by its display name and return a dataframe.

    Resolution order:
      1. Pool entry under the friendly name (group-derived, etc.) — *not*
         cached, because pool contents change at runtime when the user
         applies groups, runs Apply in Prepare, etc.
      2. Built-in reference (Guo / Zou / Luffi) loaded directly from disk —
         cached by name. read_training_source itself prefers a Prepare-loaded
         pool entry over the disk file, so changes still propagate.
      3. Custom reference from ``_dp_custom_refs.json`` — cached by path +
         mtime, so editing the source file invalidates the cache.
      4. Empty DataFrame fallback.
    """
    pool = get_prepped_pool()
    # 1. Direct pool match (mutable; never cached).
    if name in pool:
        return pool[name].copy()
    # 2. Built-in reference — cached by name.
    if name in _BUILTIN_REF_LABELS:
        df = _load_builtin_ref_cached(name)
        if df is not None and not df.empty:
            return df.copy()
    # 3. Custom reference — cached by (path, mtime).
    try:
        for cr in _dp_load_custom_refs():
            if cr.get('label') != name:
                continue
            p = Path(cr.get('path', ''))
            if not p.exists():
                return pd.DataFrame()
            try:
                _mtime = p.stat().st_mtime
            except Exception:
                _mtime = 0.0
            df = _load_custom_ref_cached(str(p), _mtime)
            return df.copy() if not df.empty else df
    except Exception:
        pass
    return pd.DataFrame()


def dataset_granularity_controls(df, key_prefix, label='Dataset'):
    """Tab-level granularity toggle: use raw samples, or one median per Group_ID.

    Renders nothing (and returns *df* unchanged) when *df* has no ``Group_ID``
    column or no group labels, so callers can invoke this unconditionally.

    Returns
    -------
    (df_to_use, granularity_label)
        ``granularity_label`` is one of ``'raw samples'`` / ``'group averages'``.
    """
    if df is None or df.empty or 'Group_ID' not in df.columns:
        return df, 'raw samples'
    _gid_present = df['Group_ID'].dropna()
    if _gid_present.empty:
        return df, 'raw samples'
    _n_raw    = int(len(df))
    _n_groups = int(_gid_present.nunique())
    _gran = st.radio(
        f'{label} granularity',
        [f'Raw samples (n = {_n_raw:,})',
         f'Group averages (n = {_n_groups} group{"s" if _n_groups != 1 else ""})'],
        horizontal=True, key=f'{key_prefix}_granularity',
        help=(
            '**Raw samples** — every row is one analysis. Use when you have many '
            'samples per group and want the model to see all of them.\n\n'
            '**Group averages** — collapse to one representative row per group '
            '(median chemistry, centroid lat/lon). The GAME-style approach: '
            'each group contributes one calibration point.'
        ),
    )
    if _gran.startswith('Group averages'):
        _tgt_cols = tuple(c for c in ('Crust_Thickness', 'Observed_km', 'Predicted_km')
                          if c in df.columns)
        _agg_df = _g_aggregate_to_groups(df, target_cols=_tgt_cols)
        return _agg_df, 'group averages'
    return df, 'raw samples'


def _multi_model_slots_ui(df, target, full_importance, default_set,
                           primary_source, default_alg, configs):
    """Render the per-group multi-model slot builder and populate *configs*.

    Each slot trains a separate ML model on a specific subset of groups, with
    its own algorithm and element list.  Granularity (raw samples vs. group
    averages) is set tab-level via ``dataset_granularity_controls`` — by the
    time *df* arrives here it is already either raw or aggregated.
    """
    _mm_key = '_mm_slots'

    # Defensive check ─────────────────────────────────────────────────────────
    if df is None or 'Group_ID' not in getattr(df, 'columns', []):
        st.info(
            'No `Group_ID` column found on this dataset. '
            'Apply groups in the **Group** tab first (then press '
            '“✓ Apply groups → …”), or switch to **Single model** above.'
        )
        return

    # Discover groups ─────────────────────────────────────────────────────────
    _grp_info: dict = {}   # gid → (display_name, n_rows)
    for _gid, _g in df.groupby('Group_ID', dropna=True):
        _gname = (
            _g['Group_Name'].dropna().iloc[0]
            if 'Group_Name' in _g.columns and _g['Group_Name'].notna().any()
            else str(_gid)
        )
        _grp_info[_gid] = (_gname, len(_g))

    if not _grp_info:
        st.info('No Group_ID values found — load a dataset with group labels from the Group tab.')
        return

    _all_gids = list(_grp_info.keys())
    _fmt_g    = lambda g: f"{_grp_info[g][0]}  (n = {_grp_info[g][1]:,})"

    # Initialise slots — default: one slot per group ──────────────────────────
    if _mm_key not in st.session_state or not isinstance(st.session_state[_mm_key], list):
        st.session_state[_mm_key] = [
            {'name': _grp_info[g][0], 'groups': [g], 'algorithm': default_alg}
            for g in _all_gids
        ]
    _mm_slots = st.session_state[_mm_key]

    st.caption(
        f'{len(_all_gids)} group(s) available · '
        f'{len(_mm_slots)} model slot(s) defined. '
        'Each slot is a **population** — a set of Group_IDs that share one '
        'trained model. A Group_ID can appear in multiple populations. '
        'With *Group averages* granularity each population trains on '
        'one median row per Group_ID; with *Raw samples*, on every analysis '
        'in the assigned groups.'
    )

    # Per-slot UI — collect (slot, feat_set, features) during render ──────────
    _delete_slot_idx = None
    _slot_render_data = []   # populated during loop, consumed by config builder

    for _si, _slot in enumerate(list(_mm_slots)):
        with st.container(border=True):
            _sc1, _sc2, _sc3 = st.columns([2.0, 3.5, 0.4])
            _slot['name'] = _sc1.text_input(
                f'Slot {_si + 1}', value=_slot.get('name', f'Model {_si + 1}'),
                key=f'mm_slot_name_{_si}', label_visibility='collapsed',
                placeholder=f'Model {_si + 1} name',
            )
            _slot['groups'] = _sc2.multiselect(
                'Groups', _all_gids,
                default=[g for g in _slot.get('groups', []) if g in _all_gids] or _all_gids[:1],
                format_func=_fmt_g,
                key=f'mm_slot_groups_{_si}', label_visibility='collapsed',
            )
            if _sc3.button('✕', key=f'mm_slot_del_{_si}', help='Remove this model slot'):
                _delete_slot_idx = _si

            normalize_widget_state(f'mm_slot_alg_{_si}', ALGORITHMS)
            _slot['algorithm'] = st.selectbox(
                'Algorithm', ALGORITHMS,
                index=(ALGORITHMS.index(_slot.get('algorithm', default_alg))
                       if _slot.get('algorithm', default_alg) in ALGORITHMS else 0),
                key=f'mm_slot_alg_{_si}',
            )

            # Per-slot element list (collapsed — inherits from last preset on first open)
            _slot_feat_set, _slot_features = feature_strategy_controls(
                f'mm_slot_{_si}', df, target, default_set, full_importance,
                collapsed=True, use_expander=True, allow_min_ri=False,
                expander_label='Element list',
            )

            # Preview: show effective training row count for this slot
            _preview_gids = _slot.get('groups', [])
            if _preview_gids:
                _n_rows = int(df['Group_ID'].isin(_preview_gids).sum())
                st.caption(
                    f'Population: **{len(_preview_gids)} Group_ID(s)** → '
                    f'**{_n_rows:,} training row(s)** · '
                    f'{len(_slot_features)} element(s)'
                )

            _slot_render_data.append((_slot, _slot_feat_set, _slot_features))

    if _delete_slot_idx is not None:
        _mm_slots.pop(_delete_slot_idx)
        st.session_state[_mm_key] = _mm_slots
        st.rerun()

    _btn1, _btn2 = st.columns(2)
    if _btn1.button('+ Add model slot', key='mm_add_slot', use_container_width=True):
        _mm_slots.append({
            'name': f'Model {len(_mm_slots) + 1}',
            'groups': [], 'algorithm': default_alg,
        })
        st.session_state[_mm_key] = _mm_slots
        st.rerun()
    if _btn2.button('↺ Reset to one-slot-per-group', key='mm_reset_slots', use_container_width=True):
        st.session_state.pop(_mm_key, None)
        st.rerun()

    st.session_state[_mm_key] = _mm_slots

    # Build configs from slots ────────────────────────────────────────────────
    _any_slot_has_features = any(feats for _, _, feats in _slot_render_data)
    if not _any_slot_has_features:
        st.info('Select at least one element in each slot\'s element list to enable training.')
        return

    for _slot, _slot_feat_set, _slot_features in _slot_render_data:
        if not _slot_features:
            continue
        _slot_gids = _slot.get('groups', [])
        if not _slot_gids:
            continue
        _slot_df = df[df['Group_ID'].isin(_slot_gids)].copy()
        if _slot_df.empty:
            continue
        _slot_label = _slot.get('name') or f'Model {_mm_slots.index(_slot) + 1}'
        configs.append({
            'label':       _slot_label,
            'df':          _slot_df,
            'target':      target,
            'features':    _slot_features,
            'feature_set': _slot_feat_set,
            'algorithm':   _slot.get('algorithm', default_alg),
            'source':      primary_source,
        })


def model_selection_controls(models, key_prefix):
    """Single/Multi-model toggle for the Validate and Predict tabs.

    Only renders a toggle when more than one model has been trained in the
    Model tab.  Returns ``(active_models, mode)`` where:

    * ``mode == 'single'``     → ``active_models`` contains exactly one entry
                                  (the user-chosen model); per-group panel is
                                  hidden by callers.
    * ``mode == 'multi'``      → ``active_models`` is the full ``models`` dict;
                                  per-group panel is available downstream.

    With 0 or 1 trained models there is nothing to choose, so the function
    returns ``(models, 'multi')`` silently — no widget is rendered, and the
    caller behaves as before.
    """
    if not models or len(models) <= 1:
        return models, 'multi'

    _mode_label = st.radio(
        'Model usage',
        ['Single model', 'Multi-model'],
        horizontal=True,
        key=f'{key_prefix}_model_mode',
        help=(
            '**Single model** — pick one of the trained models from the Model '
            'tab and apply it to every row.\n\n'
            '**Multi-model** — use all trained models. If the dataset has '
            '`Group_ID` labels, the per-group assignment panel below lets you '
            'route each group to a specific model.'
        ),
    )
    if _mode_label == 'Single model':
        _names = list(models.keys())
        _chosen = st.selectbox(
            'Trained model to use', _names,
            key=f'{key_prefix}_single_model_name',
            help='Pick which trained model is applied to the whole dataset.',
        )
        return {_chosen: models[_chosen]}, 'single'
    return models, 'multi'


def per_group_model_panel(df, tab_key, active_models, la_mode, bench_ss_key=None):
    """Render the per-group model assignment UI inside any tab.

    Appears only when *df* contains a ``Group_ID`` column (i.e. the active
    dataset was created via the Group tab's "Aggregate & save").  Each group
    gets its own model selector; clicking **▶ Run** predicts group-by-group
    and saves the merged result to the pool (and optionally to a session-state
    bench key so the Summary tab picks it up immediately).

    Parameters
    ----------
    df            : active tab dataframe (uk_raw / test_df / train_df).
    tab_key       : short string for namespacing widget + session-state keys.
    active_models : the ``models`` dict built in the Model tab.
    la_mode       : La/Yb mode string passed through to ``enrich()``.
    bench_ss_key  : optional session-state key to also write the result into.
    """
    if df is None or df.empty or 'Group_ID' not in df.columns:
        return

    # ── Persistent success banner from the most recent run ───────────────────
    _pgm_prev_key = f'_pgm_preview_{tab_key}'
    _pgm_name_key = f'_pgm_out_name_{tab_key}'
    if _pgm_prev_key in st.session_state:
        _pgm_prev_name = st.session_state.get(_pgm_name_key, '')
        _pgm_prev_df   = st.session_state[_pgm_prev_key]
        _pgm_prev_n    = len(get_prepped_pool().get(_pgm_prev_name, pd.DataFrame()))
        st.success(
            f"✓ **{_pgm_prev_name}** saved to pool ({_pgm_prev_n:,} rows — "
            f"available in all source dropdowns)."
        )
        st.dataframe(_pgm_prev_df, use_container_width=True, hide_index=True)
        if st.button('Dismiss', key=f'pgm_dismiss_{tab_key}'):
            st.session_state.pop(_pgm_prev_key, None)
            st.rerun()

    if not active_models:
        st.info('Train ML models in the **Model** tab first, then assign one per group here.')
        return

    # ── Discover groups from the active dataframe ─────────────────────────────
    _pgm_model_labels = list(active_models.keys())
    _pgm_model_opts   = ['(skip)'] + _pgm_model_labels
    _pgm_default      = _pgm_model_labels[0]
    _pgm_assign_key   = f'_pgm_assign_{tab_key}'
    _pgm_assigns      = st.session_state.setdefault(_pgm_assign_key, {})

    _pgm_groups = {}   # {gid: (gname, n_rows)}
    for _pgm_gid, _pgm_g in df.groupby('Group_ID', dropna=True):
        _pgm_gname = (
            _pgm_g['Group_Name'].dropna().iloc[0]
            if 'Group_Name' in _pgm_g.columns and _pgm_g['Group_Name'].notna().any()
            else str(_pgm_gid)
        )
        _pgm_groups[_pgm_gid] = (_pgm_gname, len(_pgm_g))

    if not _pgm_groups:
        return

    # ── One row per group: name | N | model selectbox ────────────────────────
    for _pgm_gid, (_pgm_gname, _pgm_n) in _pgm_groups.items():
        _pgm_cur = _pgm_assigns.get(str(_pgm_gid), _pgm_default)
        _pgm_cur = _pgm_cur if _pgm_cur in _pgm_model_opts else _pgm_default
        _pgm_ca, _pgm_cb = st.columns([1.4, 2.6])
        _pgm_ca.markdown(f"**{_pgm_gname}** &nbsp; *{_pgm_n:,} samples*")
        _pgm_chosen = _pgm_cb.selectbox(
            f'Model — {_pgm_gname}',
            _pgm_model_opts,
            index=_pgm_model_opts.index(_pgm_cur),
            key=f'pgm_sel_{tab_key}_{_pgm_gid}',
            label_visibility='collapsed',
        )
        _pgm_assigns[str(_pgm_gid)] = _pgm_chosen
    st.session_state[_pgm_assign_key] = _pgm_assigns

    # ── Assignment summary caption ────────────────────────────────────────────
    _pgm_used = [v for v in _pgm_assigns.values() if v != '(skip)']
    if _pgm_used:
        from collections import Counter as _PGMCounter
        _pgm_counts = _PGMCounter(_pgm_used)
        _pgm_skipped_n = sum(1 for v in _pgm_assigns.values() if v == '(skip)')
        st.caption(
            'Assignment — '
            + ' · '.join(f'**{m}**: {n} group(s)' for m, n in _pgm_counts.items())
            + (f' · *(skip)*: {_pgm_skipped_n}' if _pgm_skipped_n else '')
        )

    # ── Output name + run button ──────────────────────────────────────────────
    _pgm_out_default = f'Per-group predictions ({tab_key})'
    _pgm_c1, _pgm_c2 = st.columns([2, 1])
    _pgm_out_name = _pgm_c1.text_input(
        'Output dataset name',
        value=st.session_state.get(_pgm_name_key, _pgm_out_default),
        key=f'pgm_out_name_input_{tab_key}',
        help='Saved to the pool; immediately available in all source dropdowns.',
    )

    if _pgm_c2.button('▶ Run per-group predictions',
                      key=f'pgm_run_{tab_key}',
                      use_container_width=True):
        _pgm_rows  = []
        _pgm_warns = []
        for _pgm_gid, (_pgm_gname, _) in _pgm_groups.items():
            _pgm_assigned = _pgm_assigns.get(str(_pgm_gid), '(skip)')
            if _pgm_assigned == '(skip)' or _pgm_assigned not in active_models:
                continue
            _pgm_bundle    = active_models[_pgm_assigned]
            _pgm_grp_rows  = df[df['Group_ID'] == _pgm_gid].copy()
            if _pgm_grp_rows.empty:
                continue
            _pgm_grp_enr   = enrich(_pgm_grp_rows, la_mode, include_game=False)
            _pgm_feats     = _pgm_bundle['features']
            _pgm_missing   = [f for f in _pgm_feats if f not in _pgm_grp_enr.columns]
            if _pgm_missing:
                _pgm_warns.append(
                    f"**{_pgm_gname}**: missing features "
                    f"{_pgm_missing[:3]}{'…' if len(_pgm_missing) > 3 else ''} — skipped."
                )
                continue
            _pgm_X        = _pgm_grp_enr[_pgm_feats].apply(pd.to_numeric, errors='coerce')
            _pgm_ok       = _pgm_X.notna().all(axis=1)
            _pgm_X_ok     = _pgm_X.loc[_pgm_ok]
            if _pgm_X_ok.empty:
                _pgm_warns.append(f"**{_pgm_gname}**: no complete-feature rows — skipped.")
                continue
            _pgm_pred              = _pgm_bundle['model'].predict(_pgm_X_ok)
            _pgm_ci_lo, _pgm_ci_hi, _pgm_pred_sd = _ensemble_predict_ci(_pgm_bundle, _pgm_X_ok)
            _pgm_out               = _pgm_grp_enr.loc[_pgm_ok].copy().reset_index(drop=True)
            _pgm_out['Predicted_km']  = _pgm_pred
            _pgm_out['Model']         = _pgm_assigned
            _pgm_out['Algorithm']     = _pgm_bundle.get('algorithm', '')
            _pgm_out['Group_ID']      = _pgm_gid
            _pgm_out['Group_Name']    = _pgm_gname
            _pgm_out['Group_Source']  = 'per_group_model'
            if _pgm_ci_lo is not None:
                _pgm_out['Predicted_CI90_Low_km']   = _pgm_ci_lo
                _pgm_out['Predicted_CI90_High_km']  = _pgm_ci_hi
                _pgm_out['Predicted_SD_km']         = _pgm_pred_sd
                _pgm_out['Predicted_CI90_Width_km'] = _pgm_ci_hi - _pgm_ci_lo
            _pgm_rows.append(_pgm_out)

        for _pgm_w in _pgm_warns:
            st.warning(_pgm_w)

        if _pgm_rows:
            _pgm_result    = pd.concat(_pgm_rows, ignore_index=True)
            _pgm_save_name = _pgm_out_name.strip() or _pgm_out_default
            _pgm_pool      = get_prepped_pool()
            _pgm_pool[_pgm_save_name] = _pgm_result
            st.session_state['prepped_datasets'] = _pgm_pool
            st.session_state[_pgm_name_key]      = _pgm_save_name
            st.session_state[_pgm_prev_key]      = _pgm_result[
                [c for c in ['Group_Name', 'Model', 'Predicted_km',
                              'Predicted_CI90_Low_km', 'Predicted_CI90_High_km',
                              'Sample_ID', 'Lat', 'Lon', 'Age_Ma']
                 if c in _pgm_result.columns]
            ].head(30)
            if bench_ss_key:
                st.session_state[bench_ss_key] = _pgm_result.copy()
            st.rerun()
        else:
            st.warning(
                'No predictions produced — check that the assigned models have '
                'the required features in this dataset.'
            )


def read_training_source(source, uploaded=None):
    if source == 'From Data Prep tab':
        df = st.session_state.get('dp_training_df', pd.DataFrame())
        return df, 'Data Prep'
    if source == 'Upload dataset':
        if uploaded is None:
            return pd.DataFrame(), 'Uploaded dataset'
        return read_table(uploaded, guo_no_header=True, expected=TRAINING_COLUMNS), 'Uploaded dataset'
    # ── Prefer prepared version when one exists ──────────────────────────────
    # If the user loaded this built-in via the Prepare tab's "Load reference
    # dataset" button, the prepared dataframe lives in the pool under the
    # filename stem (e.g. 'GuoYang_2023_Model'). Use that — it has the user's
    # column-mapping decisions applied. Only fall back to the raw on-disk file
    # when no prepared version exists.
    if source in _BUILTIN_TO_FILE_STEM:
        _stem = _BUILTIN_TO_FILE_STEM[source]
        _pool_pref = st.session_state.get('prepped_datasets', {})
        if _stem in _pool_pref:
            return _pool_pref[_stem].copy(), f'{source} (prepared)'
    if source == 'Zou et al. (2021)':
        p = find_zou_training()
        if p:
            return read_table(p, guo_no_header=False), 'Zou et al. (2021)'
        # Fallback: use Guo data with Zou feature set
        p = find_training()
        if p:
            return read_table(p, guo_no_header=True, expected=TRAINING_COLUMNS), 'Guo data / Zou features'
        return pd.DataFrame(), 'Zou et al. (2021) — file not found'
    if source == 'Luffi & Ducea (2022)':
        p = find_luffi_training()
        if p:
            return _load_luffi_training(p), 'Luffi & Ducea (2022)'
        return pd.DataFrame(), 'Luffi & Ducea (2022) — file not found'
    # Pool lookup — any dataset registered in the prepped_datasets pool
    _pool = st.session_state.get('prepped_datasets', {})
    if source in _pool:
        return _pool[source].copy(), source
    # Guo & Yang (2023) default
    p = find_training()
    if p:
        return read_table(p, guo_no_header=True, expected=TRAINING_COLUMNS), 'Guo & Yang (2023)'
    return pd.DataFrame(), source

def dataset_numeric_features(df, target):
    """Return element/oxide columns present in df — excludes ratios, proxy
    estimates (H_*_km), chondrite-normalised (_star, _N), and any other
    derived column added by enrich().  Only columns in ALL_ELEMENT_FEATURES
    are returned so 'Full suite' stays clean."""
    skip = {'Sample_ID', 'Lat', 'Lon', 'Age_Ma', target, 'Crust_Thickness'}
    return [c for c in ALL_ELEMENT_FEATURES
            if c in df.columns and c not in skip and has_numeric_column(df, c)]

def bootstrap_median_convergence(values, n_bootstrap=500, ci_pct=95):
    """Bootstrap the median at logarithmically-spaced n values from 2 to len(values).
    Returns DataFrame with columns: n, median_est, ci_low, ci_high, ci_width."""
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    N = len(vals)
    if N < 3:
        return pd.DataFrame()
    ns = sorted(set(
        list(np.unique(np.round(np.geomspace(2, N, min(N - 1, 50))).astype(int))) + [N]
    ))
    alpha = (100 - ci_pct) / 2
    rng = np.random.default_rng(42)
    rows = []
    for n in ns:
        boots = np.median(rng.choice(vals, size=(n_bootstrap, n), replace=True), axis=1)
        rows.append({
            'n': int(n),
            'median_est': float(np.median(boots)),
            'ci_low': float(np.percentile(boots, alpha)),
            'ci_high': float(np.percentile(boots, 100 - alpha)),
            'ci_width': float(np.percentile(boots, 100 - alpha) - np.percentile(boots, alpha)),
        })
    return pd.DataFrame(rows)

def render_sample_size_panel(values, ci_pct=95, precision_targets=(5.0, 10.0), key_prefix='ss'):
    """Bootstrap convergence plot: how stable is the median as n grows?
    precision_targets are half-widths in km (e.g. 5 → ±5 km CI half-width)."""
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    N = len(vals)
    if N < 3:
        st.info('Need at least 3 finite predicted values for sample-size analysis.')
        return
    conv = bootstrap_median_convergence(vals, ci_pct=ci_pct)
    if conv.empty:
        return
    overall_med = float(np.median(vals))
    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(x=conv['n'], y=conv['ci_high'], mode='lines',
                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=conv['n'], y=conv['ci_low'], mode='lines',
                             line=dict(width=0), fill='tonexty',
                             fillcolor='rgba(59,130,246,0.15)',
                             name=f'{ci_pct}% CI on median', hoverinfo='skip'))
    # Median curve
    fig.add_trace(go.Scatter(x=conv['n'], y=conv['median_est'], mode='lines+markers',
                             line=dict(color='#3b82f6', width=2), marker=dict(size=4),
                             name='Bootstrap median',
                             hovertemplate='n=%{x}<br>median=%{y:.1f} km<extra></extra>'))
    # Precision threshold lines (as ±half-width bands around the full-N median)
    colors = ['#22c55e', '#f97316']
    labels = ['good', 'acceptable']
    vlines_added = set()
    for thr_half, col, lbl in zip(precision_targets, colors, labels):
        thr_full = thr_half * 2  # full CI width target
        fig.add_hrect(y0=overall_med - thr_half, y1=overall_med + thr_half,
                      fillcolor=col, opacity=0.06, line_width=0,
                      annotation_text=f'±{thr_half:.0f} km ({lbl})',
                      annotation_position='top right',
                      annotation_font=dict(size=11, color=col))
        below = conv[conv['ci_width'] <= thr_full]
        if not below.empty:
            n_cross = int(below['n'].iloc[0])
            if n_cross not in vlines_added:
                fig.add_vline(x=n_cross, line_dash='dot', line_color=col,
                              line_width=1.4,
                              annotation_text=f'n={n_cross}',
                              annotation_position='top left',
                              annotation_font=dict(color=col, size=11))
                vlines_added.add(n_cross)
    fig.update_layout(
        height=320, template='plotly_white',
        margin=dict(l=10, r=10, t=20, b=40),
        xaxis_title='Number of samples (n)',
        yaxis_title='Predicted crustal thickness (km)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    st.plotly_chart(fig, width='stretch', key=f'{key_prefix}_convergence_fig')
    # Text recommendation
    msgs = []
    for thr_half, lbl in zip(precision_targets, labels):
        below = conv[conv['ci_width'] <= thr_half * 2]
        if not below.empty:
            msgs.append(f'**{int(below["n"].iloc[0])} samples** for ±{thr_half:.0f} km ({lbl})')
        else:
            msgs.append(f'>**{N} samples** needed for ±{thr_half:.0f} km (not reached in this dataset)')
    st.caption(f'At {ci_pct}% confidence — ' + ' · '.join(msgs))

def preset_features(name, df, target):
    if name == 'Full suite':
        return dataset_numeric_features(df,target)
    features = FEATURE_SETS.get(name,FEATURE_SETS['Guo & Yang (2023)'])
    return [c for c in features if c in df]

def element_group(element):
    if element in DERIVED_RATIO_COLUMNS:
        return 'Ratios'
    groups = {
        'Major oxides':['SiO2','Al2O3','FeO','Fe2O3','Fe2O3T','MgO','CaO','Na2O','K2O','P2O5'],
        'Transition metals':['TiO2','MnO','FeO','Fe2O3','Fe2O3T','V','Cr','Co','Ni','Cu','Zn','Sc'],
        'REE':['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'],
        'HFSE':['Zr','Hf','Nb','Ta','TiO2','Th','U','Y'],
        'LILE':['Rb','Sr','Ba','K2O'],
    }
    for name, vals in groups.items():
        if element in vals:
            return name
    return 'Other trace'

COMPATIBILITY_SCORE = {
    'Cr':-1.0,'Cr/Sc':-0.95,'Ni':-0.9,'Ni/Sc':-0.85,'Sc':-0.75,'Cr/V':-0.72,'Ni/V':-0.65,'MgO':-0.55,'FeO':-0.45,'MnO':-0.35,'CaO':-0.25,'Co':-0.25,'V':-0.20,'TiO2':-0.15,
    'Gd/Yb':0.0,'Dy/Yb':0.05,'Sm/Yb':0.08,'Lu/Hf':0.10,'Nd/Yb':0.15,'Al2O3':0.18,'SiO2':0.22,'P2O5':0.25,'Y':0.30,
    'Zr/Y':0.35,'Zr/Ti':0.38,'Ce/Yb':0.42,'La/Y':0.45,'Nb/Y':0.48,'La/Sm':0.50,'La/Yb':0.55,'La/Yb(N)':0.55,'Sr/Y':0.58,'Ce/Y':0.62,'A/CaO':0.68,
    'K2O':0.72,'Hf':0.76,'Ba':0.78,'Pb':0.80,'U':0.84,'Rb':0.88,'Ba/V':0.92,'Ba/Sc':0.95,'Th/Yb':0.98,'Th':1.0,
    'La':0.55,'Ce':0.60,'Pr':0.58,'Nd':0.52,'Sm':0.45,'Eu':0.38,'Gd':0.32,'Tb':0.28,'Dy':0.24,'Ho':0.22,'Er':0.20,'Tm':0.18,'Yb':0.16,'Lu':0.14,
    'Rb_Sr':0.55,'Sr_Y':0.58,'La_Yb_raw':0.55,'La_Yb_N':0.55,'Ce_Y':0.62,'La_Y':0.45,'Dy_Yb':0.05,'Gd_Yb':0.0,'Zr_Ti':0.38,
}

def compatibility_score(feature):
    label = str(feature)
    aliases = {
        'Sr_Y':'Sr/Y','La_Yb_raw':'La/Yb','La_Yb_N':'La/Yb(N)','Ce_Y':'Ce/Y','La_Y':'La/Y','Dy_Yb':'Dy/Yb','Gd_Yb':'Gd/Yb','Zr_Ti':'Zr/Ti',
        'Rb_Sr':'Rb/Sr','Sm_Yb':'Sm/Yb','Nd_Yb':'Nd/Yb','Nb_Yb':'Nb/Yb','Th_Yb':'Th/Yb','Lu_Hf':'Lu/Hf','Ba_V':'Ba/V','Ba_Sc':'Ba/Sc','Ni_Sc':'Ni/Sc','Ni_V':'Ni/V','Cr_Sc':'Cr/Sc','Cr_V':'Cr/V',
    }
    label = aliases.get(label, label)
    if label in COMPATIBILITY_SCORE:
        return float(COMPATIBILITY_SCORE[label])
    group = element_group(str(feature))
    return {'Transition metals':-0.35,'Major oxides':0.0,'REE':0.35,'HFSE':0.65,'LILE':0.8}.get(group,0.0)

def compatibility_label(score):
    if score <= -0.35:
        return 'more compatible'
    if score >= 0.35:
        return 'more incompatible'
    return 'near-neutral'

def feature_weighting_frame(features, importance=None):
    """Build the per-feature frame the importance bar chart consumes.

    When ``importance`` carries the (optional) ``Direction`` /
    ``Signed_Importance`` columns produced by ``feature_importance``
    with df+target supplied, those are propagated through so the chart
    can render bars on a ±axis (high-feature → high-prediction goes up,
    high-feature → low-prediction goes down).  When direction is
    missing, ``Direction`` falls back to 0 and the chart degrades to
    legacy magnitude-only bars.
    """
    data = pd.DataFrame({'Feature':list(dict.fromkeys(features))})
    if data.empty:
        return data
    if importance is not None and not importance.empty and {'Feature','Relative_Importance'}.issubset(importance):
        agg_cols = ['Relative_Importance']
        if 'Direction' in importance.columns:
            agg_cols.append('Direction')
        if 'Signed_Importance' in importance.columns:
            agg_cols.append('Signed_Importance')
        imp = importance[['Feature'] + agg_cols].copy()
        for c in agg_cols:
            imp[c] = pd.to_numeric(imp[c], errors='coerce')
        # Multi-model importance frames carry one row per (Feature, Model).
        # Take the *max* magnitude row per feature so the chart represents
        # the strongest evidence; the matching Direction is read from that
        # same row to keep magnitude/direction consistent.
        imp = imp.sort_values('Relative_Importance', ascending=False)
        imp = imp.drop_duplicates(subset='Feature', keep='first')
        data = data.merge(imp, on='Feature', how='left')
    else:
        data['Relative_Importance'] = np.nan
    data['Feature_Weighting_pct'] = pd.to_numeric(data['Relative_Importance'], errors='coerce') * 100.0
    if data['Feature_Weighting_pct'].notna().sum() == 0:
        data['Feature_Weighting_pct'] = 1.0
    # Direction defaults to 0 (sign undefined) when not supplied
    if 'Direction' not in data.columns:
        data['Direction'] = 0.0
    else:
        data['Direction'] = pd.to_numeric(data['Direction'], errors='coerce').fillna(0.0)
    # Signed weighting in % — positive = high feature value pushes
    # prediction UP, negative = high feature value pushes prediction DOWN
    data['Signed_Weighting_pct'] = data['Feature_Weighting_pct'] * data['Direction']
    data['Direction_Label'] = data['Direction'].map(
        {1.0: 'high → ↑ prediction',
         -1.0: 'high → ↓ prediction',
         0.0: 'direction undefined (non-monotonic / weak)'}
    ).fillna('direction undefined')
    data['Compatibility_Score'] = data['Feature'].map(compatibility_score)
    data['Behaviour'] = data['Compatibility_Score'].map(compatibility_label)
    return data

def feature_weighting_figure(features, importance=None, sort_by='Compatibility', height=260):
    """Importance bar chart, signed when direction information is present.

    When the importance frame carries ``Direction`` (from
    ``feature_importance(..., df=, target=)``), bars use ``Signed_Weighting_pct``
    so they stretch above the zero line for "high feature value drives a
    higher prediction" and below the zero line for "high feature value
    drives a lower prediction" (i.e. **low** values are what matter).
    Bar colour still encodes geochemical compatibility (blue→red ramp)
    so the user can read direction (height) and behaviour (colour)
    independently. Features with undefined / non-monotonic direction
    show as short grey bars centred on zero.
    """
    data = feature_weighting_frame(features, importance)
    if data.empty:
        return None
    has_direction = 'Direction' in data.columns and data['Direction'].abs().sum() > 0

    # Sort either by signed magnitude (so the strongest +ve and -ve drivers
    # bookend the chart) or by compatibility (legacy left→right ramp).
    if sort_by == 'Importance':
        if has_direction:
            data = data.assign(_abs=data['Signed_Weighting_pct'].abs()).sort_values(
                ['_abs', 'Compatibility_Score'], ascending=[False, True]
            ).drop(columns='_abs')
        else:
            data = data.sort_values(['Feature_Weighting_pct','Compatibility_Score'],
                                    ascending=[False, True])
    else:
        data = data.sort_values(['Compatibility_Score','Feature_Weighting_pct'],
                                ascending=[True, False])

    if has_direction:
        # Signed axis: zero = no influence, +ve = "high → ↑", -ve = "high → ↓"
        _y_col = 'Signed_Weighting_pct'
        _y_abs_max = max(float(data[_y_col].abs().max()) * 1.25, 1.0)
        _y_range = [-_y_abs_max, _y_abs_max]
        _y_title = 'signed feature weighting (%) — +ve: high → ↑ pred · -ve: high → ↓ pred'
        # Direction undefined → render as a tiny visual nudge so it still
        # registers as a bar; otherwise zero-height bars are invisible.
        _viz_y = data[_y_col].copy()
        _viz_y[data['Direction'] == 0] = data.loc[data['Direction'] == 0,
                                                  'Feature_Weighting_pct'] * 0.05
        data = data.assign(_viz_y=_viz_y)
        _bar_y = '_viz_y'
        _hover_extra = ('<br>%{customdata[2]}<br>direction: %{customdata[3]}'
                        '<br>|importance|=%{customdata[4]:.1f}%')
        _custom = ['Behaviour', 'Compatibility_Score',
                   'Direction_Label', 'Direction', 'Feature_Weighting_pct']
    else:
        _y_col = 'Feature_Weighting_pct'
        _y_abs_max = max(float(data[_y_col].max()) * 1.25, 1.0)
        _y_range = [0, _y_abs_max]
        _y_title = 'feature weighting (%)'
        _bar_y = _y_col
        _hover_extra = ''
        _custom = ['Behaviour', 'Compatibility_Score']

    fig = px.bar(
        data, x='Feature', y=_bar_y, color='Compatibility_Score',
        color_continuous_scale=[(0.0,'#253494'),(0.48,'#f4f4f5'),(0.52,'#f4f4f5'),(1.0,'#b91c1c')],
        range_color=[-1,1], template='plotly_white',
        labels={_bar_y: _y_title,'Compatibility_Score':'compatible/incompatible'},
        custom_data=_custom,
    )
    _base_hover = '<b>%{x}</b><br>weight=%{y:.1f}%<br>%{customdata[0]}<br>compatibility score=%{customdata[1]:+.1f}'
    fig.update_traces(marker_line_color='rgba(17,24,39,0.55)', marker_line_width=0.6,
                      hovertemplate=_base_hover + _hover_extra + '<extra></extra>')
    fig.add_hline(y=0, line=dict(color='rgba(17,24,39,0.55)', width=1))
    fig.add_annotation(x=0.01,y=1.08,xref='paper',yref='paper',text='more compatible',showarrow=False,font=dict(size=10,color='#253494'),xanchor='left')
    fig.add_annotation(x=0.99,y=1.08,xref='paper',yref='paper',text='more incompatible',showarrow=False,font=dict(size=10,color='#b91c1c'),xanchor='right')
    if has_direction:
        # Tiny side-labels on the y-axis remind the reader which way is which
        fig.add_annotation(x=-0.03,y=1.0,xref='paper',yref='paper',
                           text='↑ high → ↑ pred',showarrow=False,
                           font=dict(size=9,color='#1e3a5f'),
                           xanchor='right',yanchor='top')
        fig.add_annotation(x=-0.03,y=0.0,xref='paper',yref='paper',
                           text='↓ high → ↓ pred',showarrow=False,
                           font=dict(size=9,color='#7f1d1d'),
                           xanchor='right',yanchor='bottom')
    fig.update_layout(height=height,margin=dict(l=8,r=8,t=30,b=10),coloraxis_colorbar=dict(title='',len=0.72,y=0.48,thickness=10),showlegend=False)
    fig.update_xaxes(tickangle=-65,tickfont=dict(size=9),title='')
    fig.update_yaxes(range=_y_range,title=_y_title,tickfont=dict(size=9),
                     title_font=dict(size=10),
                     gridcolor='rgba(148,163,184,0.22)',
                     zeroline=True,
                     zerolinecolor='rgba(17,24,39,0.55)',
                     zerolinewidth=1)
    return fig

def combined_feature_weighting_figure(importance_df, sort_by='Importance', height=360):
    """Combined feature-importance chart: bars = mean across models,
    markers = each model's individual value per feature.

    Why: when several models are trained side-by-side (Guo + Luffi + Zou,
    or RF + GBM + XGB), eyeballing one bar chart per model is awkward.
    This view stacks them: the bar height is the **average** importance
    across every model, and a coloured dot per model is dropped on each
    feature so you can see at a glance which models agree (dots cluster
    on the bar) and which disagree (dots scatter).

    When the importance frame carries the ``Direction`` /
    ``Signed_Importance`` columns added by ``feature_importance(..., df=, target=)``,
    everything is plotted on a signed axis so positive bars/dots mean
    "high feature value drives a higher prediction" and negative
    bars/dots mean "low feature value is what matters".
    """
    if importance_df is None or importance_df.empty:
        return None
    if not {'Feature', 'Model', 'Relative_Importance'}.issubset(importance_df.columns):
        return None

    df = importance_df.copy()
    df['Relative_Importance'] = pd.to_numeric(df['Relative_Importance'], errors='coerce')
    df['Importance_pct']      = df['Relative_Importance'] * 100.0
    has_direction = ('Direction' in df.columns
                     and pd.to_numeric(df['Direction'], errors='coerce').abs().sum() > 0)
    if has_direction:
        df['Direction']            = pd.to_numeric(df['Direction'], errors='coerce').fillna(0.0)
        df['Signed_Importance_pct'] = df['Importance_pct'] * df['Direction']
        _val_col = 'Signed_Importance_pct'
    else:
        _val_col = 'Importance_pct'

    # Aggregate: bar height = mean across models per feature
    agg = (df.groupby('Feature', as_index=False)
             .agg(_mean_val=(_val_col, 'mean'),
                  _abs_mean=(_val_col, lambda v: float(np.nanmean(np.abs(v))))))
    agg['Compatibility_Score'] = agg['Feature'].map(compatibility_score)
    agg['Behaviour']           = agg['Compatibility_Score'].map(compatibility_label)

    if sort_by == 'Importance':
        # Sort by absolute mean so the strongest +ve and -ve drivers come first
        agg = agg.sort_values('_abs_mean', ascending=False)
    else:
        agg = agg.sort_values(['Compatibility_Score', '_abs_mean'],
                              ascending=[True, False])
    feature_order = agg['Feature'].tolist()

    # Bar
    if has_direction:
        _y_abs_max = max(float(np.nanmax(np.abs(agg['_mean_val']))) * 1.35, 1.0)
        # Per-model dots can extend further than the mean bars, so widen
        # the y-range to fit them too.
        _per_model_max = float(np.nanmax(np.abs(df[_val_col]))) * 1.10
        _y_abs_max = max(_y_abs_max, _per_model_max, 1.0)
        _y_range = [-_y_abs_max, _y_abs_max]
        _y_title = 'mean signed importance (%) — bars = average across models · dots = each model'
    else:
        _y_abs_max = max(float(np.nanmax(agg['_mean_val'])) * 1.35, 1.0)
        _y_per_model_max = float(np.nanmax(df[_val_col])) * 1.10
        _y_abs_max = max(_y_abs_max, _y_per_model_max, 1.0)
        _y_range = [0, _y_abs_max]
        _y_title = 'mean importance (%) — bars = average across models · dots = each model'

    fig = px.bar(
        agg, x='Feature', y='_mean_val', color='Compatibility_Score',
        color_continuous_scale=[(0.0,'#253494'),(0.48,'#f4f4f5'),(0.52,'#f4f4f5'),(1.0,'#b91c1c')],
        range_color=[-1, 1], template='plotly_white',
        labels={'_mean_val': _y_title, 'Compatibility_Score': 'compatible/incompatible'},
        custom_data=['Behaviour', 'Compatibility_Score', '_abs_mean'],
        category_orders={'Feature': feature_order},
    )
    fig.update_traces(marker_line_color='rgba(17,24,39,0.55)', marker_line_width=0.6,
                      opacity=0.55,
                      hovertemplate='<b>%{x}</b><br>mean=%{y:+.1f}%<br>|mean|=%{customdata[2]:.1f}%'
                                    '<br>%{customdata[0]}<br>compatibility score=%{customdata[1]:+.1f}<extra></extra>',
                      name='mean across models', showlegend=False)

    # Per-model dots — distinct colour per model from a categorical palette,
    # symbol per algorithm so RF / GBM / XGB stay visually separable when
    # the same feature set is trained with multiple algorithms.
    model_palette = (px.colors.qualitative.Bold + px.colors.qualitative.Set2
                     + px.colors.qualitative.Pastel)
    models_in_view = list(dict.fromkeys(df['Model'].astype(str).tolist()))
    for mi, mname in enumerate(models_in_view):
        sub = df[df['Model'] == mname]
        # Reindex to feature_order so each model's dot lines up with its
        # bar; missing features get NaN and are silently dropped by Plotly.
        sub = sub.set_index('Feature').reindex(feature_order).reset_index()
        fig.add_trace(go.Scatter(
            x=sub['Feature'], y=sub[_val_col], mode='markers',
            marker=dict(symbol='circle', size=10,
                        color=model_palette[mi % len(model_palette)],
                        line=dict(color='rgba(17,24,39,0.8)', width=1.0)),
            name=mname,
            customdata=np.c_[sub[_val_col].fillna(0).to_numpy()],
            hovertemplate=(f'<b>{mname}</b><br>%{{x}}: %{{y:+.2f}}%<extra></extra>'),
        ))

    fig.add_hline(y=0, line=dict(color='rgba(17,24,39,0.55)', width=1))
    if has_direction:
        fig.add_annotation(x=-0.03,y=1.0,xref='paper',yref='paper',
                           text='↑ high → ↑ pred',showarrow=False,
                           font=dict(size=9,color='#1e3a5f'),
                           xanchor='right',yanchor='top')
        fig.add_annotation(x=-0.03,y=0.0,xref='paper',yref='paper',
                           text='↓ high → ↓ pred',showarrow=False,
                           font=dict(size=9,color='#7f1d1d'),
                           xanchor='right',yanchor='bottom')
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=30, b=10),
                      coloraxis_colorbar=dict(title='', len=0.72, y=0.48, thickness=10),
                      legend=dict(orientation='h', y=-0.32, x=0,
                                  font=dict(size=9), bgcolor='rgba(255,255,255,0.6)'),
                      showlegend=True)
    fig.update_xaxes(tickangle=-65, tickfont=dict(size=9), title='',
                     categoryorder='array', categoryarray=feature_order)
    fig.update_yaxes(range=_y_range, title=_y_title,
                     tickfont=dict(size=9), title_font=dict(size=10),
                     gridcolor='rgba(148,163,184,0.22)',
                     zeroline=True, zerolinecolor='rgba(17,24,39,0.55)', zerolinewidth=1)
    return fig


def render_element_chips(features):
    if not features:
        st.info('No elements selected.')
        return
    colors = {
        'Major oxides':'#eef2ff',
        'Transition metals':'#eff6ff',
        'REE':'#ecfdf5',
        'HFSE':'#fff7ed',
        'LILE':'#fdf2f8',
        'Ratios':'#fef9c3',
        'Other trace':'#f3f4f6',
    }
    for group in ['Major oxides','Transition metals','LILE','HFSE','REE','Ratios','Other trace']:
        vals = [f for f in features if element_group(f)==group]
        if not vals:
            continue
        chips = ''.join([f'<span style="display:inline-block;margin:0 6px 6px 0;padding:4px 9px;border-radius:999px;background:{colors[group]};border:1px solid #d1d5db;font-size:0.86rem;">{v}</span>' for v in vals])
        st.markdown(f'**{group}**<br>{chips}',unsafe_allow_html=True)

IONIC_FIELDS = {
    'LFS / LIL': [(0.95,1.82),(2.35,1.42),(2.35,1.20),(0.95,1.12)],
    'REE': [(2.65,1.18),(4.15,1.00),(4.00,0.84),(2.65,0.88)],
    'HFS': [(2.95,1.22),(6.0,0.86),(5.45,0.72),(3.0,0.72)],
}
IONIC_POINTS = {
    'Li':(1,0.92),'Na2O':(1,1.18),'K2O':(1,1.52),'Rb':(1,1.61),'Cs':(1,1.74),'Sr':(2,1.26),'Ba':(2,1.42),'Pb':(2,1.19),
    'MgO':(2,0.72),'CaO':(2,1.00),'FeO':(2,0.78),'Fe2O3':(3,0.65),'Fe2O3T':(3,0.65),'MnO':(2,0.83),'Co':(2,0.75),'Ni':(2,0.69),'Cu':(2,0.73),'Zn':(2,0.74),'Cr':(3,0.62),'V':(3,0.64),'Sc':(3,0.75),
    'Al2O3':(3,0.54),'SiO2':(4,0.40),'TiO2':(4,0.75),'P2O5':(5,0.38),
    'Y':(3,1.02),'La':(3,1.16),'Ce':(3,1.14),'Pr':(3,1.13),'Nd':(3,1.11),'Sm':(3,1.08),'Eu':(3,1.07),'Eu2+':(2,1.25),'Eu3+':(3,1.07),'Gd':(3,1.05),'Tb':(3,1.04),'Dy':(3,1.03),'Ho':(3,1.02),'Er':(3,1.00),'Tm':(3,0.99),'Yb':(3,0.99),'Lu':(3,0.98),
    'Zr':(4,0.84),'Hf':(4,0.83),'Nb':(5,0.74),'Ta':(5,0.74),'Th':(4,1.05),'U':(6,0.87),
}

def render_ionic_diagram(features, key='ionic_diagram'):
    pts = []
    for f in features:
        if f == 'Eu':
            pts.extend([
                {'Element':'Eu2+','Charge':IONIC_POINTS['Eu2+'][0],'Radius_A':IONIC_POINTS['Eu2+'][1],'Group':'LILE'},
                {'Element':'Eu3+','Charge':IONIC_POINTS['Eu3+'][0],'Radius_A':IONIC_POINTS['Eu3+'][1],'Group':'REE'},
            ])
        elif f in IONIC_POINTS:
            pts.append({'Element':f,'Charge':IONIC_POINTS[f][0],'Radius_A':IONIC_POINTS[f][1],'Group':element_group(f)})
    if not pts:
        st.info('No selected elements have ionic radius coordinates for this diagram.')
        return
    fig = go.Figure()
    field_colors = {'LFS / LIL':'rgba(34,197,94,0.08)','REE':'rgba(52,211,153,0.08)','HFS':'rgba(234,179,8,0.08)'}
    for name, poly in IONIC_FIELDS.items():
        xs=[p[0] for p in poly]+[poly[0][0]]
        ys=[p[1] for p in poly]+[poly[0][1]]
        fig.add_trace(go.Scatter(x=xs,y=ys,fill='toself',mode='lines',name=name,line=dict(color='rgba(17,24,39,0.95)',width=1.2),fillcolor=field_colors[name],hoverinfo='skip',showlegend=False))
    band_x=np.array([0.75,1.35,2.2,3.2,4.2,5.15,5.45,4.7,3.8,2.85,1.8,0.8])
    band_y=np.array([1.50,1.42,1.26,1.05,0.74,0.35,0.35,0.62,0.86,1.08,1.30,1.40])
    fig.add_trace(go.Scatter(x=band_x,y=band_y,fill='toself',mode='lines',line=dict(width=0),fillcolor='rgba(167,243,208,0.45)',hoverinfo='skip',showlegend=False))
    ip_x=np.linspace(1.65,3.9,60)
    fig.add_trace(go.Scatter(x=ip_x,y=ip_x/2,mode='lines',line=dict(color='rgba(75,85,99,0.75)',width=1.35,dash='dash'),hovertemplate='Ionic potential = charge/radius = 2.0<extra></extra>',showlegend=False))
    plot=pd.DataFrame(pts)
    plot['Z_over_r']=plot['Charge']/plot['Radius_A']
    ree_members=['La','Ce','Pr','Nd','Sm','Eu3+','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
    selected_ree=plot[plot['Element'].isin(ree_members)].copy()
    main_plot=plot[~plot['Element'].isin(ree_members)].copy()
    if not selected_ree.empty:
        ree_hover='<br>'.join([f"{r.Element}: {r.Radius_A:.1f} A, Z/r {r.Z_over_r:.1f}" for r in selected_ree.itertuples(index=False)])
        ree_marker=pd.DataFrame([{'Element':'REE group','Charge':3.0,'Radius_A':1.05,'Group':'REE','Z_over_r':selected_ree['Z_over_r'].mean(),'Hover_List':ree_hover}])
        main_plot=pd.concat([main_plot,ree_marker],ignore_index=True)
    else:
        main_plot['Hover_List']=''
    if 'Hover_List' not in main_plot:
        main_plot['Hover_List']=''
    else:
        main_plot['Hover_List']=main_plot['Hover_List'].fillna('')
    fig.add_trace(go.Scatter(
        x=main_plot['Charge'],y=main_plot['Radius_A'],mode='markers',
        marker=dict(size=np.where(main_plot['Element'].eq('REE group'),14,9),color=main_plot['Z_over_r'],colorscale='Turbo',showscale=True,colorbar=dict(title='Z/r',len=0.72,y=0.50),line=dict(color='black',width=0.8),opacity=0.96),
        customdata=np.c_[main_plot['Element'],main_plot['Group'],main_plot['Z_over_r'],main_plot['Hover_List']],
        hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[3]}<br>Z/r=%{customdata[2]:.1f}<br>Charge=%{x:.0f}<br>Radius=%{y:.1f} A<extra></extra>',
        name='Selected elements'
    ))
    label_set={'K2O','Rb','Ba','Sr','Eu2+','Eu3+','Y','La','Lu','Th','U','Ce','Zr','Hf','Nb','Ta','TiO2','P2O5','SiO2','Al2O3','MgO','FeO','MnO','CaO'}
    label_offsets={'K2O':(-16,-14),'Rb':(-15,0),'Ba':(13,-4),'Sr':(13,0),'Eu2+':(13,8),'Eu3+':(13,-8),'La':(-16,-8),'Lu':(13,8),'Y':(-14,0),'Ce':(12,0),'Th':(10,-8),'U':(12,0),'Zr':(12,0),'Hf':(-14,0),'Nb':(0,-14),'Ta':(0,-14),'SiO2':(0,-14),'Al2O3':(0,14),'P2O5':(0,-14)}
    for row in main_plot.itertuples(index=False):
        if row.Element not in label_set:
            continue
        ax,ay=label_offsets.get(row.Element,(10,-10))
        fig.add_annotation(x=row.Charge,y=row.Radius_A,text=row.Element,showarrow=False,xshift=ax,yshift=ay,font=dict(size=10,color='rgba(17,24,39,0.95)'))
    if not selected_ree.empty:
        inset_x=5.45
        y_min,y_max=1.22,1.72
        fig.add_trace(go.Scatter(x=[inset_x,inset_x],y=[y_min,y_max],mode='lines',line=dict(color='black',width=1),hoverinfo='skip',showlegend=False))
        for yv in np.arange(y_min,y_max+0.001,0.04):
            fig.add_trace(go.Scatter(x=[inset_x-0.03,inset_x+0.03],y=[yv,yv],mode='lines',line=dict(color='black',width=0.6),hoverinfo='skip',showlegend=False))
        inset_rank={el:i for i,el in enumerate(ree_members)}
        selected_ree['Inset_Y']=[y_max-(inset_rank.get(el,0)/(max(len(ree_members)-1,1)))*(y_max-y_min) for el in selected_ree['Element']]
        fig.add_trace(go.Scatter(
            x=np.full(len(selected_ree),inset_x+0.10),y=selected_ree['Inset_Y'],mode='markers',
            marker=dict(size=5,color=selected_ree['Z_over_r'],colorscale='Turbo',cmin=plot['Z_over_r'].min(),cmax=plot['Z_over_r'].max(),line=dict(color='black',width=0.5),showscale=False),
            customdata=np.c_[selected_ree['Element'],selected_ree['Z_over_r'],selected_ree['Radius_A']],
            hovertemplate='<b>%{customdata[0]}</b><br>REE inset<br>Z/r=%{customdata[1]:.1f}<br>Radius=%{customdata[2]:.1f} A<extra></extra>',
            showlegend=False
        ))
        label_side={'La':0.18,'Ce':0.36,'Pr':0.18,'Nd':0.36,'Sm':0.18,'Eu3+':0.36,'Gd':0.18,'Tb':0.36,'Dy':0.18,'Ho':0.36,'Er':0.18,'Tm':0.36,'Yb':0.18,'Lu':0.36}
        for row in selected_ree.itertuples(index=False):
            fig.add_annotation(x=inset_x+label_side.get(row.Element,0.18),y=row.Inset_Y,text=row.Element.replace('Eu3+','Eu'),showarrow=False,font=dict(size=9,color='rgba(17,24,39,0.95)'),xanchor='left')
        fig.add_annotation(x=inset_x-0.08,y=y_max+0.02,text='LREEs',showarrow=False,textangle=-90,font=dict(size=9,color='rgba(17,24,39,0.95)'))
        fig.add_annotation(x=inset_x-0.08,y=y_min+0.02,text='HREEs',showarrow=False,textangle=-90,font=dict(size=9,color='rgba(17,24,39,0.95)'))
        fig.add_annotation(x=inset_x+0.23,y=y_min-0.05,text='3+',showarrow=False,font=dict(size=11,color='rgba(17,24,39,0.95)'))
    fig.add_annotation(x=4.55,y=1.68,text='Incompatible<br>elements',showarrow=False,font=dict(size=14,color='rgba(17,24,39,0.95)'))
    fig.add_annotation(x=1.65,y=0.50,text='Compatible<br>elements',showarrow=False,font=dict(size=13,color='rgba(17,24,39,0.95)'))
    fig.add_annotation(x=4.75,y=1.00,text='HFS',showarrow=False,font=dict(size=15,color='rgba(17,24,39,0.95)'))
    fig.add_annotation(x=3.35,y=1.04,text='REE',showarrow=False,font=dict(size=13,color='rgba(17,24,39,0.95)'))
    fig.add_annotation(x=1.38,y=1.55,text='LFS, LIL',showarrow=False,font=dict(size=13,color='rgba(17,24,39,0.95)'))
    fig.add_annotation(x=3.18,y=1.59,text='Ionic potential = 2.0',textangle=-66,showarrow=False,font=dict(size=10,color='rgba(75,85,99,0.9)'),bgcolor='rgba(255,255,255,0.55)',borderpad=1)
    fig.update_layout(height=470,template='plotly_white',margin=dict(l=10,r=10,t=20,b=8),xaxis_title='Ionic charge',yaxis_title='Ionic radius (A)',showlegend=False)
    fig.update_xaxes(range=[0.5,6.5],dtick=1,showline=True,linecolor='black',mirror=False)
    fig.update_yaxes(range=[0.25,1.9],showline=True,linecolor='black',mirror=False)
    st.plotly_chart(fig,width='stretch',key=key)

def render_element_summary(features, key='element_summary', full_importance=None):
    chips_col, chart_col = st.columns([0.72,1.68])
    with chips_col:
        render_element_chips(features)
    with chart_col:
        weight_fig = feature_weighting_figure(features, full_importance, sort_by='Compatibility', height=230)
        if weight_fig is not None:
            st.plotly_chart(weight_fig,width='stretch',key=f'{key}_weights')
        render_ionic_diagram(features,key=f'{key}_ionic')

def feature_strategy_controls(prefix, df, target, default_set, full_importance=None, collapsed=True, use_expander=True, allow_min_ri=True, expander_label=None):
    panel = st.expander(expander_label or f'{prefix} elements', expanded=not collapsed) if use_expander else st.container()
    with panel:
        strategies = ['Preset','Custom list'] + (['Minimum RI'] if allow_min_ri else [])
        strategy = st.radio(f'{prefix} element strategy',strategies,horizontal=True,key=f'{prefix}_strategy')
        if strategy == 'Preset':
            preset_options = ['Guo & Yang (2023)','Zou et al. (2021)','Luffi & Ducea (2022)','Full suite','Immobile elements']
            preset = st.selectbox(f'{prefix} preset',preset_options,index=preset_options.index(default_set) if default_set in preset_options else 0,key=f'{prefix}_preset')
            features = preset_features(preset,df,target)
            # Stash preset name + its features so Custom list can inherit them
            st.session_state[f'{prefix}_last_preset_name']     = preset
            st.session_state[f'{prefix}_last_preset_features'] = [c for c in features if c in df.columns]
            label = preset
        elif strategy == 'Minimum RI':
            min_ri_pct = typed_slider(st,f'{prefix} minimum full-model RI (%)',0.0,10.0,0.5,0.1,key=f'{prefix}_ri')
            features = threshold_feature_set(full_importance if full_importance is not None else pd.DataFrame(), min_ri_pct)
            label = f'RI >= {min_ri_pct:.1f}%'
            st.caption(f'{len(features)} features selected')
        else:
            numeric_options = dataset_numeric_features(df, target)
            # Seed the multiselect from the last active preset, but only when the
            # preset changes — preserves any additions the user made in Custom mode.
            _last_preset  = st.session_state.get(f'{prefix}_last_preset_name', default_set)
            _last_feats   = st.session_state.get(
                f'{prefix}_last_preset_features',
                [c for c in preset_features(default_set, df, target) if c in numeric_options],
            )
            _seeded_from  = st.session_state.get(f'{prefix}_custom_seeded_from')
            if _seeded_from != _last_preset:
                # First switch to Custom (or preset changed) — seed from the preset
                st.session_state[f'{prefix}_custom']            = [c for c in _last_feats if c in numeric_options]
                st.session_state[f'{prefix}_custom_seeded_from'] = _last_preset
            features = st.multiselect(
                f'{prefix} elements', numeric_options,
                key=f'{prefix}_custom',
            )
            label = 'Custom'

        # ── Optional: add individual ratios ──────────────────────────────────
        # Ratios are derived columns (Sr_Y, La_Yb_N, etc.) — excluded from the
        # base element list but individually available here.
        _avail_ratios = [c for c in DERIVED_RATIO_COLUMNS if c in df.columns and c not in features]
        if _avail_ratios:
            _ratio_key = f'{prefix}_extra_ratios'
            _ratio_labels = {
                'Sr_Y':'Sr/Y', 'La_Yb_raw':'La/Yb (raw)', 'La_Yb_N':'La/Yb(N)',
                'Ce_Y':'Ce/Y', 'Zr_Y':'Zr/Y', 'Dy_Yb':'Dy/Yb', 'Gd_Yb':'Gd/Yb',
                'Sm_Yb':'Sm/Yb', 'Ce_Yb':'Ce/Yb', 'Nd_Yb':'Nd/Yb', 'Nb_Yb':'Nb/Yb',
                'Th_Yb':'Th/Yb', 'La_Y':'La/Y', 'Nd_Y':'Nd/Y', 'Nb_Y':'Nb/Y',
                'Th_Y':'Th/Y', 'La_Sm':'La/Sm', 'Ba_V':'Ba/V', 'Ba_Sc':'Ba/Sc',
                'Ni_Sc':'Ni/Sc', 'Ni_V':'Ni/V', 'Cr_Sc':'Cr/Sc', 'Cr_V':'Cr/V',
                'Lu_Hf':'Lu/Hf', 'Zr_Ti':'Zr/Ti', 'Rb_Sr':'Rb/Sr',
            }
            with st.expander('Add element ratios (optional)', expanded=False):
                st.caption(
                    'Element ratios are computed from raw measurements. '
                    'Select any you want included as additional model features.'
                )
                _selected_ratios = st.multiselect(
                    'Ratios to include',
                    _avail_ratios,
                    default=st.session_state.get(_ratio_key, []),
                    format_func=lambda c: _ratio_labels.get(c, c.replace('_', '/')),
                    key=_ratio_key,
                )
            features = features + [r for r in _selected_ratios if r not in features]
        else:
            _selected_ratios = []

        missing = [c for c in features if c not in df]
        if missing:
            st.warning(f'Missing columns: {", ".join(missing)}')
        render_element_summary([c for c in features if c in df], key=f'{prefix}_element_summary', full_importance=full_importance)
    return label, [c for c in features if c in df]

def training_subset_controls(prefix, df, expanded=False, target=None, noun='training'):
    if df.empty:
        return df
    filtered = enrich(df, la_mode)  # enrich() copies internally; no caller copy needed
    with st.expander(f'{prefix} {noun} filters', expanded=expanded):
        if 'Age_Ma' in filtered:
            ages = pd.to_numeric(filtered['Age_Ma'], errors='coerce')
            finite = ages[np.isfinite(ages)]
            if not finite.empty:
                min_age = float(np.nanmax([0, np.floor(finite.min())]))
                max_age = float(np.ceil(finite.max()))
                if max_age > min_age:
                    age_range = typed_range_slider(st, f'{prefix} age range (Ma)', min_age, max_age, (min_age, max_age), key=f'{prefix}_age_range')
                    filtered = filtered.loc[ages.between(age_range[0], age_range[1], inclusive='both')]
                else:
                    st.caption(f'{prefix} age range: {min_age:.0f} Ma')
        thickness_col = target if target in filtered else target_col(filtered)
        if thickness_col in filtered:
            thickness = pd.to_numeric(filtered[thickness_col], errors='coerce')
            finite_thickness = thickness[np.isfinite(thickness)]
            if not finite_thickness.empty:
                min_thick = float(np.floor(finite_thickness.min()))
                max_thick = float(np.ceil(finite_thickness.max()))
                if max_thick > min_thick:
                    thick_range = typed_range_slider(st, f'{prefix} crustal thickness range (km)', min_thick, max_thick, (min_thick, max_thick), key=f'{prefix}_thickness_range')
                    filtered = filtered.loc[thickness.between(thick_range[0], thick_range[1], inclusive='both')]
                else:
                    st.caption(f'{prefix} crustal thickness: {min_thick:.0f} km')
        rock_options = [r for r in ['ultramafic','mafic','intermediate','felsic','unclassified'] if 'Rock_Type_Model' in filtered and filtered['Rock_Type_Model'].eq(r).any()]
        if rock_options:
            selected_rocks = st.multiselect(f'{prefix} rock types', rock_options, default=rock_options, key=f'{prefix}_rocks')
            filtered = filtered[filtered['Rock_Type_Model'].isin(selected_rocks)]
        age_cat_cols = [c for c in ['Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label'] if c in filtered]
        if age_cat_cols:
            age_cat_col = st.selectbox(f'{prefix} geological time category', age_cat_cols, index=min(1,len(age_cat_cols)-1), key=f'{prefix}_geo_time_col')
            age_cats = sorted([str(v) for v in filtered[age_cat_col].dropna().unique()])
            if age_cats:
                selected_age_cats = st.multiselect(f'{prefix} geological time values', age_cats, default=age_cats, key=f'{prefix}_geo_time_values')
                filtered = filtered[filtered[age_cat_col].astype(str).isin(selected_age_cats)]
        setting_cols = [c for c in ['Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset'] if c in filtered]
        if setting_cols:
            setting_col = st.selectbox(f'{prefix} tectonic/grouping column', setting_cols, index=0, key=f'{prefix}_setting_col')
            settings = sorted([str(v) for v in filtered[setting_col].dropna().unique()])
            if settings:
                selected_settings = st.multiselect(f'{prefix} tectonic/grouping values', settings, default=settings, key=f'{prefix}_settings')
                filtered = filtered[filtered[setting_col].astype(str).isin(selected_settings)]
        st.caption(f'{len(filtered)} of {len(df)} rows selected for {noun}')
    return filtered

def robust_sigma(values):
    s = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    if len(s) < 2:
        return np.nan
    med = float(s.median())
    mad = float(np.median(np.abs(s - med)))
    if mad > 0:
        return 1.4826 * mad
    return float(s.std(ddof=1))

def age_window_mask(age_series, target_age, age_window):
    ages = pd.to_numeric(age_series, errors='coerce')
    if pd.isna(target_age) or age_window is None:
        return pd.Series(True, index=ages.index)
    return ages.notna() & ages.sub(float(target_age)).abs().le(float(age_window))

def local_error_envelope(plot, model_name, xs, slope=None, intercept=None, method='Window', window_km=10.0, min_n=10, use_age_window=False, age_window=10.0):
    g = plot[plot['Model'].astype(str).eq(str(model_name))].copy() if 'Model' in plot else plot.copy()
    obs = pd.to_numeric(g['Observed_km'], errors='coerce')
    pred = pd.to_numeric(g['Predicted_km'], errors='coerce')
    fit = (slope * obs + intercept) if slope is not None and intercept is not None else obs
    resid = pred - fit
    valid = obs.notna() & resid.notna()
    if valid.sum() < 2:
        return np.full(len(xs), np.nan)
    global_sigma = robust_sigma(resid[valid])
    out = []
    for x in xs:
        age_mask = pd.Series(True, index=g.index)
        if use_age_window and 'Age_Ma' in g:
            nearest_age_pool = g.loc[valid].assign(_dist=obs[valid].sub(float(x)).abs()).sort_values('_dist')
            if not nearest_age_pool.empty:
                target_age = pd.to_numeric(nearest_age_pool['Age_Ma'], errors='coerce').dropna()
                target_age = target_age.iloc[0] if not target_age.empty else np.nan
                age_mask = age_window_mask(g['Age_Ma'], target_age, age_window)
        if method == 'Global':
            vals = resid[valid & age_mask]
            if len(vals) < min_n:
                vals = resid[valid]
        else:
            local = valid & age_mask & obs.sub(float(x)).abs().le(window_km / 2)
            vals = resid[local] if local.sum() >= min_n else resid[valid]
        sigma = robust_sigma(vals)
        out.append(float(sigma if np.isfinite(sigma) else global_sigma))
    return np.asarray(out, dtype=float)

def benchmark_figure(df, point_size=6, color_by='Model', show_fit=True, show_point_error=True, show_error_envelope=True, error_method='Window', window_km=10.0, min_n=10, use_age_window=False, age_window=10.0, show_tree_ci=False, show_moving_avg=False, moving_avg_n=20, moving_avg_type='Median', show_points=True):
    plot = plot_df(df.copy())
    plot['Delta_km'] = np.round(plot['Predicted_km'] - plot['Observed_km'], 1)
    plot['Abs_Residual_km'] = np.round(np.abs(plot['Delta_km']), 1)
    plot['Delta_percent'] = np.round(np.where(plot['Observed_km']!=0,100*plot['Delta_km']/plot['Observed_km'],np.nan), 1)
    has_ci = show_tree_ci and {'Predicted_CI90_Low_km','Predicted_CI90_High_km'}.issubset(plot.columns)
    if has_ci:
        plot['_CI_upper_err'] = pd.to_numeric(plot['Predicted_CI90_High_km'], errors='coerce') - pd.to_numeric(plot['Predicted_km'], errors='coerce')
        plot['_CI_lower_err'] = pd.to_numeric(plot['Predicted_km'], errors='coerce') - pd.to_numeric(plot['Predicted_CI90_Low_km'], errors='coerce')
        plot['_CI_upper_err'] = plot['_CI_upper_err'].clip(lower=0)
        plot['_CI_lower_err'] = plot['_CI_lower_err'].clip(lower=0)
        error_arg = '_CI_upper_err'
        error_minus_arg = '_CI_lower_err'
    else:
        error_arg = 'Abs_Residual_km' if show_point_error else None
        error_minus_arg = None
    color_col = color_by if color_by in plot else 'Model'
    scatter_kwargs = {}
    if is_residual_layer(color_col):
        scatter_kwargs.update(color_continuous_scale=RESIDUAL_COLORSCALE, color_continuous_midpoint=0)
    ci_sd_cols = [c for c in ['Predicted_CI90_Low_km','Predicted_CI90_High_km','Predicted_SD_km','Predicted_CI90_Width_km'] if c in plot]
    custom_cols = ['Delta_km','Delta_percent','Model','Abs_Residual_km'] + ci_sd_cols
    fig=px.scatter(
        plot,x='Observed_km',y='Predicted_km',color=color_col,template='plotly_white',
        custom_data=custom_cols,
        error_y=error_arg,
        **(dict(error_y_minus=error_minus_arg) if error_minus_arg else {}),
        labels={
            'Observed_km':'known: crustal thickness [Km]',
            'Predicted_km':'model: crustal thickness [Km]',
            'Abs_Residual_km':'absolute residual [Km]',
            'Delta_km':'model - known [Km]',
            'Model':'Model',
            'Algorithm':'Algorithm',
        },
        **scatter_kwargs,
    )
    # Count scatter traces before any manual add_trace; assign per-model colours
    # that match the Plotly qualitative cycle used by px.scatter.
    n_px_traces = len(fig.data)
    _plotly_colors = px.colors.qualitative.Plotly
    _model_order = [str(m) for m in plot['Model'].unique()]
    _model_color = {m: _plotly_colors[i % len(_plotly_colors)] for i, m in enumerate(_model_order)}
    # Rename scatter traces so the legend reads "Model X (points)" and optionally
    # hide the scatter layer while keeping legend entries visible for identification.
    for trace in fig.data[:n_px_traces]:
        if getattr(trace, 'name', None):
            trace.name = str(trace.name) + ' (points)'
        if not show_points:
            trace.visible = 'legendonly'
    ci_hover = ('<br>90% CI: %{customdata[4]:.1f}–%{customdata[5]:.1f} Km'
                '<br>SD: %{customdata[6]:.1f} Km'
                '<br>CI width: %{customdata[7]:.1f} Km') if has_ci and len(ci_sd_cols)==4 else ''
    fig.update_traces(
        marker=dict(size=point_size,line=dict(color='black',width=0.7)),
        error_y=dict(thickness=0.7,width=2,color='rgba(100,100,100,0.55)'),
        hovertemplate=(
            'known: crustal thickness=%{x:.1f} Km<br>'
            'model: crustal thickness=%{y:.1f} Km<br>'
            'model - known=%{customdata[0]:+.1f} Km<br>'
            'absolute residual=%{customdata[3]:.1f} Km<br>'
            'residual=%{customdata[1]:+.1f}%<br>'
            '%{customdata[2]}' + ci_hover + '<extra></extra>'
        )
    )
    fig.add_trace(go.Scatter(x=[0,90],y=[0,90],mode='lines',name='1:1',line=dict(color='black',dash='dash')))
    if show_fit:
        for name,g in plot.groupby('Model'):
            clean_fit=g[['Observed_km','Predicted_km']].apply(pd.to_numeric,errors='coerce').dropna()
            if len(clean_fit)<2:
                continue
            slope,intercept=np.polyfit(clean_fit['Observed_km'],clean_fit['Predicted_km'],1)
            xs_dense=np.linspace(0,90,181)
            ys_dense=slope*xs_dense+intercept
            line_col = _model_color.get(str(name), '#888888')
            if show_error_envelope:
                sigma_curve = local_error_envelope(plot,name,xs_dense,slope,intercept,error_method,window_km,min_n,use_age_window,age_window)
                for label,z,opacity in [('99%',2.576,0.035),('95%',1.96,0.07)]:
                    envelope = z * sigma_curve
                    if np.isfinite(envelope).any():
                        fig.add_trace(go.Scatter(
                            x=np.r_[xs_dense,xs_dense[::-1]], y=np.r_[ys_dense+envelope,(ys_dense-envelope)[::-1]],
                            fill='toself', mode='lines', line=dict(width=0), opacity=opacity,
                            name=f'{name} best-fit {label} residual envelope',
                            hoverinfo='skip'
                        ))
            xs=np.arange(0,95,5)
            ys=slope*xs+intercept
            delta=np.round(ys-xs, 1)
            pct=np.round(np.divide(100*delta,xs,out=np.full_like(delta,np.nan,dtype=float),where=xs!=0), 1)
            # Solid, thick line — draws on top of scatter after the trace re-order below
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', name=f'{name} best fit',
                customdata=np.c_[delta,pct],
                line=dict(color=line_col, width=2.5),
                hovertemplate='known: crustal thickness=%{x:.1f} Km<br>model best fit=%{y:.1f} Km<br>model - known=%{customdata[0]:+.1f} Km<br>%{customdata[1]:+.1f}%<extra></extra>',
            ))
    # ── Moving average — independent of show_fit ─────────────────────────────
    if show_moving_avg:
        _n = max(2, int(moving_avg_n))
        _ma_fn = 'median' if str(moving_avg_type).lower() == 'median' else 'mean'
        _ma_label_suffix = f'moving {_ma_fn} (n={_n})'
        for name, g in plot.groupby('Model'):
            line_col = _model_color.get(str(name), '#888888')
            ma_src = (g[['Observed_km','Predicted_km']]
                      .apply(pd.to_numeric, errors='coerce')
                      .sort_values('Observed_km')
                      .dropna())
            if len(ma_src) < 2:
                continue
            roll = ma_src['Predicted_km'].rolling(_n, min_periods=max(1, _n // 2))
            ma_y = roll.median() if _ma_fn == 'median' else roll.mean()
            valid_ma = ma_y.notna()
            if valid_ma.sum() < 2:
                continue
            x_vals = ma_src['Observed_km'][valid_ma].values
            y_vals = ma_y[valid_ma].values
            # Interpolate to a regular 5 km grid for marker points
            xs_grid = np.arange(0, 91, 5)
            x_min, x_max = float(x_vals.min()), float(x_vals.max())
            in_range = (xs_grid >= x_min) & (xs_grid <= x_max)
            ma_grid_x = xs_grid[in_range]
            ma_grid_y = np.interp(ma_grid_x, x_vals, y_vals) if len(ma_grid_x) > 0 else np.array([])
            _full_label = f'{name} {_ma_label_suffix}'
            # Smooth line (dense)
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                name=_full_label,
                line=dict(color=line_col, width=2.5),
                opacity=0.85,
                hovertemplate=_full_label + '<br>known=%{x:.1f} km<br>predicted=%{y:.1f} km<extra></extra>',
            ))
            # Markers at every 5 km (separate trace, shares legend entry)
            if len(ma_grid_x) > 0:
                fig.add_trace(go.Scatter(
                    x=ma_grid_x, y=ma_grid_y,
                    mode='markers',
                    name=_full_label,
                    showlegend=False,
                    marker=dict(size=8, color=line_col, symbol='circle',
                                line=dict(color='white', width=1.2)),
                    hoverinfo='skip',
                ))
    # ── Re-order traces: 1:1 + envelopes below scatter; lines above scatter ──
    px_traces  = fig.data[:n_px_traces]
    extra      = fig.data[n_px_traces:]
    one_one    = extra[:1]                  # 1:1 reference (always first add_trace)
    rest_extra = extra[1:]
    envelopes  = tuple(t for t in rest_extra if getattr(t, 'fill', None) == 'toself')
    toplines   = tuple(t for t in rest_extra if getattr(t, 'fill', None) != 'toself')
    # Draw order: 1:1 → envelopes → scatter points → best-fit → moving avg
    fig.data = one_one + envelopes + px_traces + toplines
    fig.update_layout(
        legend=dict(
            groupclick='togglegroup',
            orientation='h',
            yanchor='top',
            y=-0.18,
            xanchor='left',
            x=0,
            font=dict(size=11),
        ),
        margin=dict(b=130),
    )
    return fig

def cv_frame(clean, target, features, model_name, seed=42, algorithm='ExtraTrees'):
    algorithm = display_algorithm_label(algorithm)
    model_name = display_algorithm_label(model_name)
    y,p,r2,rmse = cv(clean,target,features,seed,algorithm)
    return pd.DataFrame({'Observed_km':y,'Predicted_km':p,'Model':model_name,'Algorithm':algorithm,'R2':r2,'RMSE_km':rmse})

def train_model_set(train_df, target, names, seed=42, custom_sets=None, algorithms=None):
    feature_lookup = dict(FEATURE_SETS)
    if custom_sets:
        feature_lookup.update(custom_sets)
    algorithms = [display_algorithm_label(a) for a in (algorithms or ['ExtraTrees'])]
    models = {}
    validation = []
    importance = []
    for name in names:
        features = feature_lookup[name]
        for algorithm in algorithms:
            label = f'{name} / {algorithm}'
            model, clean = train_model(train_df,target,features,seed,algorithm)
            models[label] = {'model':model,'clean':clean,'features':features,'feature_set':name,'algorithm':algorithm}
            validation.append(cv_frame(clean,target,features,label,seed,algorithm))
            fi = feature_importance(model,features, df=clean, target=target)
            if not fi.empty:
                fi.insert(0,'Model',label)
                fi.insert(1,'Feature_Set',name)
                fi.insert(2,'Algorithm',algorithm)
                importance.append(fi)
    val_df = pd.concat(validation,ignore_index=True) if validation else pd.DataFrame()
    imp_df = pd.concat(importance,ignore_index=True) if importance else pd.DataFrame()
    return models, val_df, imp_df

def _sanitise_model_name(name) -> str:
    """Convert a model registry key (e.g. 'Guo & Yang (2023) / ExtraTrees')
    into a column-safe suffix (e.g. 'Guo_Yang_2023_ExtraTrees'). Strips
    punctuation, collapses runs of underscores, trims edges. Used to build
    per-model wide columns in the validation / prediction bench."""
    import re as _re
    s = str(name).strip()
    s = _re.sub(r'[^0-9A-Za-z]+', '_', s)
    s = _re.sub(r'_+', '_', s).strip('_')
    return s or 'model'


def _ensemble_predict_ci(bundle, X_df, ci_lo=5, ci_hi=95):
    """Per-tree prediction CI for bagging ensembles (ExtraTrees, RandomForest).
    Returns (ci_low, ci_high, pred_sd) arrays or (None, None, None) for boosting models."""
    pipe = bundle['model']
    X = X_df.to_numpy(dtype=float)
    # Walk through all pipeline steps except the last, applying transforms
    if hasattr(pipe, 'named_steps'):
        steps = list(pipe.named_steps.items())
        X_t = X.copy()
        for _, transformer in steps[:-1]:
            X_t = transformer.transform(X_t)
        estimator = steps[-1][1]
    else:
        X_t = X
        estimator = pipe
    # Requires a flat list of fitted decision trees (ExtraTrees / RandomForest)
    estimators = getattr(estimator, 'estimators_', None)
    if not isinstance(estimators, list) or len(estimators) == 0:
        return None, None, None
    # (n_trees, n_samples) matrix of predictions
    tree_preds = np.array([t.predict(X_t) for t in estimators], dtype=float)
    return (
        np.percentile(tree_preds, ci_lo, axis=0),
        np.percentile(tree_preds, ci_hi, axis=0),
        tree_preds.std(axis=0),
    )

@st.cache_data(show_spinner=False)
def benchmark_uploaded(_models, test_df, target, seed=42, la_yb_mode='raw_ppm'):
    rows = []
    test_df = enrich(test_df.copy(), la_yb_mode, include_game=True)
    proxy_thickness_cols = list(PROXY_THICKNESS_LABELS) if 'PROXY_THICKNESS_LABELS' in globals() else ['H_Profeta2015_SrY_km','H_Sundell2021_SrY_km','H_Zou2021_SrY_SVRE_km','H_Profeta2015_LaYbN_km','H_Sundell2021_LaYbN_km','H_Zou2021_LaYbN_SVRE_km','H_Mantle2008_CeY_sample_km','H_Sundell2021_Paired_km','H_GAME_LuffiDucea2022_km']
    proxy_value_cols = list(PROXY_VALUE_LIBRARY) if 'PROXY_VALUE_LIBRARY' in globals() else ['Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2']
    game_cols = ['GAME_Luffi2022_N_mohometers','GAME_Luffi2022_N_raw_mohometers','GAME_Luffi2022_MAD_km','GAME_Luffi2022_IQR_km','GAME_Luffi2022_CI95_Low_km','GAME_Luffi2022_CI95_High_km','GAME_Luffi2022_CI95_Width_km','GAME_Luffi2022_Reliability','GAME_Luffi2022_Status']
    _meta_priority = ['Sample_ID','Lat','Lon','Age_Ma','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type','Rock_Type_Model','MgO','SiO2',target]
    meta_cols = [c for c in _meta_priority + proxy_value_cols + proxy_thickness_cols + game_cols if c in test_df]
    # Pass through ALL remaining columns from the upload so the Group tab
    # can see Country, Province, Terrane, and any other user-supplied columns.
    _meta_set = set(meta_cols)
    meta_cols += [c for c in test_df.columns if c not in _meta_set]
    for name,bundle in _models.items():
        features = bundle['features']
        need = features + [target]
        missing = [c for c in need if c not in test_df]
        if missing:
            continue
        clean = test_df[list(dict.fromkeys(need + meta_cols))].copy()
        numeric_need = clean[need].apply(pd.to_numeric,errors='coerce')
        clean = clean.loc[numeric_need.notna().all(axis=1)].copy()
        if clean.empty:
            continue
        X_numeric = clean[features].apply(pd.to_numeric, errors='coerce')
        pred = bundle['model'].predict(X_numeric)
        ci_lo, ci_hi, pred_sd = _ensemble_predict_ci(bundle, X_numeric)
        tmp = clean[list(dict.fromkeys([c for c in meta_cols if c != target]))].reset_index(drop=True)
        tmp['Observed_km'] = pd.to_numeric(clean[target], errors='coerce').values.ravel()
        tmp['Known_Thickness_km'] = tmp['Observed_km']
        tmp['Predicted_km'] = pred
        tmp['Model'] = name
        tmp['Algorithm'] = bundle.get('algorithm','')
        tmp['Residual_km'] = tmp['Predicted_km'] - tmp['Observed_km']
        if ci_lo is not None:
            tmp['Predicted_CI90_Low_km']   = ci_lo
            tmp['Predicted_CI90_High_km']  = ci_hi
            tmp['Predicted_SD_km']         = pred_sd
            tmp['Predicted_CI90_Width_km'] = ci_hi - ci_lo
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    long_bench = ensure_unique_columns(pd.concat(rows, ignore_index=True))
    # Per-model wide columns — surface each trained model's predictions as
    # their own column so the Validation plot picker can plot ExtraTrees vs
    # RandomForest directly (instead of every row having a single
    # Predicted_km column whose meaning depends on the Model column).
    # Each sample row gets EVERY model's column filled; same value
    # replicates across the rows that share a Sample_ID so the picker can
    # consume the long-format bench unchanged.
    long_bench = _attach_per_model_wide_columns(long_bench)
    return long_bench


def _attach_per_model_wide_columns(long_bench):
    """Add per-model wide columns (Predicted_<safe>_km, Residual_<safe>_km,
    Predicted_CI90_Low_<safe>_km, Predicted_CI90_High_<safe>_km,
    Predicted_CI90_Width_<safe>_km, Predicted_SD_<safe>_km) to a long-format
    validation / prediction bench. Uses Sample_ID when present, otherwise
    falls back to a positional sample key derived from the per-model row
    order. Each sample's row block (one row per model) ends up carrying
    every model's per-model columns — values replicate across the block."""
    if long_bench.empty or 'Model' not in long_bench.columns:
        return long_bench
    if 'Sample_ID' in long_bench.columns and long_bench['Sample_ID'].notna().any():
        _sample_key = '_pm_sample_key'
        long_bench = long_bench.copy()
        long_bench[_sample_key] = long_bench['Sample_ID'].astype(str)
    else:
        # Fallback: each model's rows are written in the same input order,
        # so the per-model cumulative row index identifies the same sample
        # across model blocks.
        _sample_key = '_pm_sample_key'
        long_bench = long_bench.copy()
        long_bench[_sample_key] = long_bench.groupby('Model').cumcount().astype(int).astype(str)
    _pm_cols = [c for c in ['Predicted_km', 'Residual_km',
                            'Predicted_CI90_Low_km', 'Predicted_CI90_High_km',
                            'Predicted_CI90_Width_km', 'Predicted_SD_km']
                if c in long_bench.columns]
    def _to_per_model(col, safe):
        # Insert _<safe> immediately before the trailing _km. Falls back
        # to suffixing for columns that don't end in _km.
        if col.endswith('_km'):
            return f'{col[:-3]}_{safe}_km'
        return f'{col}_{safe}'
    for _mn in long_bench['Model'].dropna().astype(str).unique():
        _safe = _sanitise_model_name(_mn)
        _sub = long_bench.loc[long_bench['Model'].astype(str) == _mn,
                              [_sample_key] + _pm_cols].drop_duplicates(subset=[_sample_key])
        _sub = _sub.rename(columns={c: _to_per_model(c, _safe) for c in _pm_cols})
        long_bench = long_bench.merge(_sub, on=_sample_key, how='left')
    long_bench = long_bench.drop(columns=[_sample_key], errors='ignore')
    return ensure_unique_columns(long_bench)

@st.cache_data(show_spinner=False)
def predict_uploaded(_models, pred_df, seed=42, la_yb_mode='raw_ppm'):
    rows = []
    pred_df = enrich(pred_df.copy(), la_yb_mode, include_game=True)
    proxy_thickness_cols = list(PROXY_THICKNESS_LABELS)
    proxy_value_cols = list(PROXY_VALUE_LIBRARY)
    game_cols = ['GAME_Luffi2022_N_mohometers','GAME_Luffi2022_N_raw_mohometers','GAME_Luffi2022_MAD_km','GAME_Luffi2022_IQR_km','GAME_Luffi2022_CI95_Low_km','GAME_Luffi2022_CI95_High_km','GAME_Luffi2022_CI95_Width_km','GAME_Luffi2022_Reliability','GAME_Luffi2022_Status']
    _meta_priority = ['Sample_ID','Lat','Lon','Age_Ma','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type','Rock_Type_Model','MgO','SiO2']
    meta_cols = [c for c in _meta_priority + proxy_value_cols + proxy_thickness_cols + game_cols if c in pred_df]
    # Pass through ALL remaining columns from the upload so the Group tab
    # can see Country, Province, Terrane, and any other user-supplied columns.
    _meta_set = set(meta_cols)
    meta_cols += [c for c in pred_df.columns if c not in _meta_set]
    for name, bundle in _models.items():
        features = bundle['features']
        if any(c not in pred_df for c in features):
            continue
        cols_needed = list(dict.fromkeys(features + [c for c in meta_cols if c in pred_df]))
        clean = pred_df[cols_needed].copy()
        X_numeric = clean[features].apply(pd.to_numeric, errors='coerce')
        ok = X_numeric.notna().all(axis=1)
        clean = clean.loc[ok].copy()
        X_numeric = X_numeric.loc[ok]
        if clean.empty:
            continue
        pred = bundle['model'].predict(X_numeric)
        ci_lo, ci_hi, pred_sd = _ensemble_predict_ci(bundle, X_numeric)
        tmp = clean[[c for c in meta_cols if c in clean]].reset_index(drop=True)
        tmp['Predicted_km'] = pred
        tmp['Model'] = name
        tmp['Algorithm'] = bundle.get('algorithm', '')
        if ci_lo is not None:
            tmp['Predicted_CI90_Low_km'] = ci_lo
            tmp['Predicted_CI90_High_km'] = ci_hi
            tmp['Predicted_SD_km'] = pred_sd
            tmp['Predicted_CI90_Width_km'] = ci_hi - ci_lo
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    long_bench = ensure_unique_columns(pd.concat(rows, ignore_index=True))
    # Surface per-model wide columns (see _attach_per_model_wide_columns for
    # rationale) so the Predict tab's Validation plot picker can address
    # individual trained ML models.
    return _attach_per_model_wide_columns(long_bench)

def prediction_summary_stats(df):
    rows = []
    if df.empty or not {'Model','Algorithm','Predicted_km'}.issubset(df):
        return pd.DataFrame()
    for (model, algorithm), g in df.groupby(['Model','Algorithm']):
        p = pd.to_numeric(g['Predicted_km'], errors='coerce').dropna()
        if p.empty:
            continue
        rows.append({
            'Model': model, 'Algorithm': algorithm, 'N': len(p),
            'Median_km': p.median(), 'Mean_km': p.mean(),
            'SD_km': p.std(), 'Q25_km': p.quantile(0.25),
            'Q75_km': p.quantile(0.75), 'Min_km': p.min(), 'Max_km': p.max(),
        })
    return tidy_numbers(pd.DataFrame(rows))

def benchmark_readiness_report(models, test_df, target):
    rows = []
    for name,bundle in models.items():
        features = bundle.get('features', [])
        required = list(dict.fromkeys(features + ([target] if target else [])))
        missing = [c for c in required if c not in test_df]
        complete_rows = 0
        rows_total = len(test_df)
        rows_with_target = 0
        rows_with_all_features = 0
        top_blockers = []
        if not missing and required:
            numeric = test_df[required].apply(pd.to_numeric,errors='coerce')
            if target in numeric:
                rows_with_target = int(numeric[target].notna().sum())
            feature_cols = [c for c in features if c in numeric]
            if feature_cols:
                rows_with_all_features = int(numeric[feature_cols].notna().all(axis=1).sum())
            complete_rows = int(numeric.notna().all(axis=1).sum())
            missing_counts = numeric.isna().sum().sort_values(ascending=False)
            top_blockers = [f'{c}: {int(n)} missing/non-numeric' for c,n in missing_counts.items() if n > 0][:10]
        else:
            top_blockers = [f'{c}: missing column' for c in missing[:15]]
        rows.append({
            'Model':name,
            'Rows':rows_total,
            'Rows_with_target':rows_with_target,
            'Rows_with_all_features':rows_with_all_features,
            'Complete_rows':complete_rows,
            'Missing_required_columns':', '.join(missing) if missing else '',
            'Missing_feature_count':len([c for c in features if c not in test_df]),
            'Top_blockers':'; '.join(top_blockers),
            'Reason':('missing required columns' if missing else 'no rows complete for required numeric fields' if complete_rows == 0 else 'ready'),
        })
    return pd.DataFrame(rows)

def benchmark_failure_message(report):
    """Compose a human-readable explanation of why benchmarking returned 0 rows.

    Two cases:
      • Some/all models have ``Complete_rows == 0`` — list the
        per-model reason + top blockers (this is the common case).
      • Every model has ``Complete_rows > 0`` — readiness disagrees
        with the empty bench, which typically means the bench shown
        is from an older run made before the active models were
        trained/swapped. Tell the user to re-run rather than emit a
        nonsense "everything looks fine" line.
    """
    if report.empty:
        return 'No benchmark readiness information is available.'
    parts = []
    for r in report.itertuples(index=False):
        reason = getattr(r,'Reason','')
        blockers = getattr(r,'Top_blockers','')
        missing = getattr(r,'Missing_required_columns','')
        complete = getattr(r,'Complete_rows',0)
        if complete:
            continue
        detail = blockers or missing or reason
        parts.append(f'{r.Model}: {reason}. {detail}'.strip())
    if parts:
        return '\n'.join(parts[:5])
    return ('Readiness check says rows are available, but the bench shown '
            'is from a previous run (likely before the current models were '
            'trained or selected). Press **▶ Run validation** to recompute.')

def validation_summary(df):
    rows = []
    if df.empty or not {'Model','Algorithm','Observed_km','Predicted_km'}.issubset(df):
        return pd.DataFrame()
    for (model, algorithm), g in df.groupby(['Model','Algorithm']):
        y = pd.to_numeric(g['Observed_km'], errors='coerce')
        p = pd.to_numeric(g['Predicted_km'], errors='coerce')
        mask = y.notna() & p.notna()
        if not mask.any():
            continue
        resid = p[mask] - y[mask]
        rows.append({
            'Model': model,
            'Algorithm': algorithm,
            'n': int(mask.sum()),
            'R2': r2_score(y[mask], p[mask]) if mask.sum() > 1 else np.nan,
            'RMSE_km': float(np.sqrt(np.mean(resid**2))),
            'MAE_km': float(np.mean(np.abs(resid))),
            'Bias_km': float(np.mean(resid)),
        })
    return tidy_numbers(pd.DataFrame(rows))

def display_validation_summary(df):
    summary = validation_summary(df)
    if summary.empty:
        return summary
    return summary.rename(columns={
        'n':'Samples',
        'R2':'R2',
        'RMSE_km':'RMSE [Km]',
        'MAE_km':'MAE [Km]',
        'Bias_km':'Bias [Km]',
    })

def div(a,b):
    a = pd.to_numeric(a,errors='coerce'); b = pd.to_numeric(b,errors='coerce')
    return np.where((a.notna())&(b.notna())&(b!=0),a/b,np.nan)

def pln(s):
    s = pd.to_numeric(s,errors='coerce'); return np.where(s>0,np.log(s),np.nan)

def game_div(out, numerator, denominator):
    if numerator not in out or denominator not in out:
        return pd.Series(np.nan, index=out.index, dtype='float64')
    return pd.Series(div(out[numerator], out[denominator]), index=out.index)

def game_series(out, sensor):
    """Return the y-axis series for a GAME sensor.

    IMPORTANT: the Luffi & Ducea (2022) GAME calibration file uses RAW
    elemental ratios (medianed per elevation bin), NOT chondrite-normalised
    ones — verified by direct comparison of the file's La/Yb, Ce/Yb, Nd/Yb,
    Sm/Yb, Gd/Yb, Dy/Yb, La/Y, Lu/Hf columns to median(El_num)/median(El_den)
    of the same rows: ratio(listed/raw) ≈ 1.00 across the board. So this
    function returns RAW ratios for every sensor, matching the calibration.

    The (La/Yb)N chondrite-normalised value lives in `out['La_Yb_N']` and is
    used by the separate Profeta 2015 / Sundell 2021 / Zou 2021 / Hu 2017
    LaYbN proxies (see `H_*_LaYbN_km` columns), NOT by GAME.
    """
    nan = pd.Series(np.nan, index=out.index, dtype='float64')
    def col(name):
        return pd.to_numeric(out[name], errors='coerce') if name in out else nan
    ratio_cols = {
        'LaYb':'La_Yb_raw','CeYb':'Ce_Yb','NdYb':'Nd_Yb','LaY':'La_Y','LuHf':'Lu_Hf',
        'SmYb':'Sm_Yb','NdY':'Nd_Y','NbY':'Nb_Y','CeY':'Ce_Y','LaSm':'La_Sm',
        'ThYb':'Th_Yb','GdYb':'Gd_Yb','ZrY':'Zr_Y','BaV':'Ba_V','BaSc':'Ba_Sc',
        'NbYb':'Nb_Yb','ThY':'Th_Y','DyYb':'Dy_Yb','NiSc':'Ni_Sc','CrSc':'Cr_Sc',
        'NiV':'Ni_V','CrV':'Cr_V','ZrTi':'Zr_Ti'
    }
    if sensor in ratio_cols and ratio_cols[sensor] in out:
        return col(ratio_cols[sensor])
    if sensor == 'LaYb': return game_div(out,'La','Yb')
    if sensor == 'CeYb': return game_div(out,'Ce','Yb')
    if sensor == 'NdYb': return game_div(out,'Nd','Yb')
    if sensor == 'LaY': return game_div(out,'La','Y')
    if sensor == 'LuHf': return game_div(out,'Lu','Hf')
    if sensor == 'SmYb': return game_div(out,'Sm','Yb')
    if sensor == 'NdY': return game_div(out,'Nd','Y')
    if sensor == 'NbY': return game_div(out,'Nb','Y')
    if sensor == 'CeY': return game_div(out,'Ce','Y')
    if sensor == 'LaSm': return game_div(out,'La','Sm')
    if sensor == 'ThYb': return game_div(out,'Th','Yb')
    if sensor == 'GdYb': return game_div(out,'Gd','Yb')
    if sensor == 'ZrY': return game_div(out,'Zr','Y')
    if sensor == 'BaV': return game_div(out,'Ba','V')
    if sensor == 'BaSc': return game_div(out,'Ba','Sc')
    if sensor == 'NbYb': return game_div(out,'Nb','Yb')
    if sensor == 'ThY': return game_div(out,'Th','Y')
    if sensor == 'DyYb': return game_div(out,'Dy','Yb')
    if sensor == 'NiSc': return game_div(out,'Ni','Sc')
    if sensor == 'CrSc': return game_div(out,'Cr','Sc')
    if sensor == 'NiV': return game_div(out,'Ni','V')
    if sensor == 'CrV': return game_div(out,'Cr','V')
    if sensor == 'ZrTi':
        if {'Zr','TiO2'}.issubset(out):
            return pd.Series(div(out['Zr'], pd.to_numeric(out['TiO2'], errors='coerce') * 5995.1), index=out.index)
        return nan
    if sensor == 'A':
        return col('Na2O') + col('K2O') if {'Na2O','K2O'}.issubset(out) else nan
    if sensor == 'ACaO':
        if {'Na2O','K2O','CaO'}.issubset(out):
            return pd.Series(div(col('Na2O') + col('K2O'), out['CaO']), index=out.index)
        return nan
    if sensor == 'SrYx':
        sr_y = game_div(out,'Sr','Y')
        rb_sr = game_div(out,'Rb','Sr')
        return sr_y.where(rb_sr.between(0.05,0.2,inclusive='both'))
    if sensor == 'FeOt':
        if 'FeO' in out:
            return col('FeO')
        if 'Fe2O3T' in out:
            return col('Fe2O3T') * FE2O3_TO_FEO
        if 'Fe2O3' in out:
            return col('Fe2O3') * FE2O3_TO_FEO
        return nan
    return col(sensor)

@st.cache_data(show_spinner=False)
def load_game_calibration():
    candidates = [
        Path(__file__).resolve().parent / GAME_CALIBRATION_FILE,
        Path.cwd() / GAME_CALIBRATION_FILE,
        GAME_CALIBRATION_FILE,
    ]
    for path in candidates:
        if path.exists():
            return ensure_unique_columns(pd.read_csv(path))
    return pd.DataFrame()

def _game_loess_predict_one(cal, target_xy):
    xy = cal['xy']
    z = cal['z']
    n = len(z)
    if n < 6 or cKDTree is None:
        return np.nan
    k = min(max(12, int(np.ceil(0.35 * n))), n)
    dists, idx = cal['tree'].query(target_xy, k=k)
    idx = np.atleast_1d(idx)
    dists = np.atleast_1d(dists).astype(float)
    local_xy = xy[idx]
    local_z = z[idx]
    max_dist = float(np.nanmax(dists)) if len(dists) else 0.0
    if not np.isfinite(max_dist) or max_dist <= 0:
        weights = np.ones(len(idx))
    else:
        u = np.clip(dists / max_dist, 0, 1)
        weights = (1 - u**3)**3
        weights = np.where(weights <= 1e-9, 1e-9, weights)
    dx = local_xy[:,0] - target_xy[0]
    dy = local_xy[:,1] - target_xy[1]
    design = np.column_stack([np.ones(len(idx)), dx, dy, dx*dy])
    root_w = np.sqrt(weights)
    try:
        beta, *_ = np.linalg.lstsq(design * root_w[:,None], local_z * root_w, rcond=None)
        return float(beta[0])
    except Exception:
        return float(np.average(local_z, weights=weights))

def _game_loocv_rmse(xy, z):
    if cKDTree is None or len(z) < 12:
        return np.nan
    tree = cKDTree(xy)
    preds = []
    actual = []
    n = len(z)
    k = min(max(13, int(np.ceil(0.35 * n)) + 1), n)
    for i, row in enumerate(xy):
        dists, idx = tree.query(row, k=k)
        idx = np.atleast_1d(idx)
        keep = idx != i
        idx = idx[keep]
        if len(idx) < 6:
            continue
        sub = {'xy':xy[idx], 'z':z[idx], 'tree':cKDTree(xy[idx])}
        pred = _game_loess_predict_one(sub, row)
        if np.isfinite(pred):
            preds.append(pred)
            actual.append(z[i])
    if not preds:
        return np.nan
    return float(np.sqrt(np.mean((np.array(preds) - np.array(actual))**2)))

@st.cache_resource(show_spinner=False)
def game_calibrators():
    cal_df = load_game_calibration()
    calibrators = {}
    if cal_df.empty or cKDTree is None:
        return calibrators
    mgo_col = 'MgO (wt%)'
    elev_col = 'elevation (km)'
    if mgo_col not in cal_df or elev_col not in cal_df:
        return calibrators
    mgo = pd.to_numeric(cal_df[mgo_col], errors='coerce')
    elev = pd.to_numeric(cal_df[elev_col], errors='coerce')
    for sensor, label, calib_col in GAME_SENSORS:
        if calib_col not in cal_df:
            continue
        y = pd.to_numeric(cal_df[calib_col], errors='coerce')
        mask = mgo.notna() & y.notna() & elev.notna()
        if int(mask.sum()) < 12:
            continue
        raw_xy = np.column_stack([mgo[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)])
        z = elev[mask].to_numpy(dtype=float)
        center = np.nanmedian(raw_xy, axis=0)
        scale = np.nanstd(raw_xy, axis=0)
        scale = np.where((~np.isfinite(scale)) | (scale == 0), 1.0, scale)
        xy = (raw_xy - center) / scale
        hull = None
        if Delaunay is not None and len(xy) >= 4:
            try:
                hull = Delaunay(xy)
            except QhullError:
                hull = None
        calibrators[sensor] = {
            'label':label,
            'calib_col':calib_col,
            'xy':xy,
            'raw_xy':raw_xy,
            'z':z,
            'center':center,
            'scale':scale,
            'tree':cKDTree(xy),
            'hull':hull,
            'rmse_elev':_game_loocv_rmse(xy,z),
        }
    return calibrators

def _bootstrap_ci_median(values, seed=42, n_boot=250):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    picks = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
    meds = np.nanmedian(picks, axis=1)
    return (float(np.nanpercentile(meds, 2.5)), float(np.nanpercentile(meds, 97.5)))

def apply_luffi_game_mohometers(df, alpha=GAME_ALPHA_DEFAULT, beta=GAME_BETA_DEFAULT, rmse_max_elev=2.0, focus_mad_km=5.0, final_mad_km=10.0, min_mohometers=3, seed=42):
    out = df.copy()
    out['H_GAME_LuffiDucea2022_km'] = np.nan
    out['GAME_Luffi2022_N_mohometers'] = 0
    out['GAME_Luffi2022_N_raw_mohometers'] = 0
    out['GAME_Luffi2022_MAD_km'] = np.nan
    out['GAME_Luffi2022_IQR_km'] = np.nan
    out['GAME_Luffi2022_CI95_Low_km'] = np.nan
    out['GAME_Luffi2022_CI95_High_km'] = np.nan
    out['GAME_Luffi2022_CI95_Width_km'] = np.nan
    out['GAME_Luffi2022_Reliability'] = 'not run'
    out['GAME_Luffi2022_Status'] = 'not run'
    calibrators = game_calibrators()
    if not calibrators:
        out['GAME_Luffi2022_Status'] = 'GAME calibration unavailable'
        out['GAME_Luffi2022_Reliability'] = 'No estimate'
        return out
    if 'MgO' not in out:
        out['GAME_Luffi2022_Status'] = 'missing MgO'
        out['GAME_Luffi2022_Reliability'] = 'No estimate'
        return out
    mgo = pd.to_numeric(out['MgO'], errors='coerce')
    estimates = pd.DataFrame(index=out.index)
    used_sensors = []
    for sensor, label, _ in GAME_SENSORS:
        cal = calibrators.get(sensor)
        if cal is None:
            continue
        rmse_elev = cal.get('rmse_elev', np.nan)
        if np.isfinite(rmse_elev) and rmse_elev > rmse_max_elev:
            continue
        y = game_series(out, sensor)
        valid = mgo.notna() & y.notna()
        values = np.full(len(out), np.nan)
        if valid.any():
            raw = np.column_stack([mgo[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float)])
            scaled = (raw - cal['center']) / cal['scale']
            in_domain = np.ones(len(scaled), dtype=bool)
            if cal.get('hull') is not None:
                try:
                    in_domain = cal['hull'].find_simplex(scaled) >= 0
                except Exception:
                    in_domain = np.ones(len(scaled), dtype=bool)
            local_preds = np.array([_game_loess_predict_one(cal, row) if domain_ok else np.nan for row, domain_ok in zip(scaled, in_domain)], dtype=float)
            values[np.where(valid)[0]] = alpha * local_preds + beta
        if np.isfinite(values).any():
            estimates[f'GAME_{sensor}_H_km'] = values
            used_sensors.append(label)
    if estimates.empty:
        out['GAME_Luffi2022_Status'] = 'no usable mohometers'
        out['GAME_Luffi2022_Reliability'] = 'No estimate'
        return out
    # Write individual sensor columns into out so the coverage bar chart can read them.
    for _s_col, _s_vals in estimates.items():
        out[_s_col] = _s_vals.values
    # Vectorized: compute into plain arrays, assign to DataFrame once at the end.
    n_rows = len(out)
    est_arr = estimates.to_numpy(dtype=float)  # shape (n_rows, n_sensors)
    _game_H      = np.full(n_rows, np.nan)
    _game_N      = np.zeros(n_rows, int)
    _game_N_raw  = np.zeros(n_rows, int)
    _game_MAD    = np.full(n_rows, np.nan)
    _game_IQR    = np.full(n_rows, np.nan)
    _game_CIlo   = np.full(n_rows, np.nan)
    _game_CIhi   = np.full(n_rows, np.nan)
    _game_CIw    = np.full(n_rows, np.nan)
    _game_rel    = np.full(n_rows, 'No estimate', dtype=object)
    _game_stat   = np.full(n_rows, 'no usable mohometers', dtype=object)
    for pos in range(n_rows):
        row_vals = est_arr[pos]
        vals = row_vals[np.isfinite(row_vals)]
        _game_N_raw[pos] = len(vals)
        if len(vals) < min_mohometers:
            _game_N[pos] = len(vals)
            _game_stat[pos] = f'low N ({len(vals)} mohometers)'
            _game_rel[pos] = 'Low N'
            continue
        raw_median = float(np.nanmedian(vals))
        dev = np.abs(vals - raw_median)
        keep = dev <= focus_mad_km
        status = 'focus MAD'
        if int(keep.sum()) < min_mohometers:
            keep = dev <= final_mad_km
            status = 'final MAD'
        if int(keep.sum()) < min_mohometers:
            keep = np.isfinite(vals)
            status = 'all valid; high spread'
        kept = vals[keep]
        if len(kept) < min_mohometers:
            _game_N[pos] = len(kept)
            _game_stat[pos] = 'low N after MAD filtering'
            _game_rel[pos] = 'Low N'
            continue
        q25, q75 = np.nanpercentile(kept, [25, 75])
        row_seed = seed + pos
        ci_low, ci_high = _bootstrap_ci_median(kept, seed=row_seed)
        mad = float(np.nanmedian(np.abs(kept - np.nanmedian(kept))))
        ci_width = float(ci_high - ci_low) if np.isfinite(ci_low) and np.isfinite(ci_high) else np.nan
        if len(kept) < 8 or mad > 10 or (np.isfinite(ci_width) and ci_width > 15) or status == 'all valid; high spread':
            reliability = 'Low confidence'
        elif len(kept) < 15 or mad > 5 or (np.isfinite(ci_width) and ci_width > 8):
            reliability = 'Caution'
        else:
            reliability = 'Good'
        _game_H[pos]    = float(np.nanmedian(kept))
        _game_N[pos]    = len(kept)
        _game_MAD[pos]  = mad
        _game_IQR[pos]  = float(q75 - q25)
        _game_CIlo[pos] = ci_low
        _game_CIhi[pos] = ci_high
        _game_CIw[pos]  = ci_width
        _game_rel[pos]  = reliability
        _game_stat[pos] = f'{status}; {len(kept)} of {len(vals)} mohometers'
    out['H_GAME_LuffiDucea2022_km']      = _game_H
    out['GAME_Luffi2022_N_mohometers']   = _game_N
    out['GAME_Luffi2022_N_raw_mohometers'] = _game_N_raw
    out['GAME_Luffi2022_MAD_km']         = _game_MAD
    out['GAME_Luffi2022_IQR_km']         = _game_IQR
    out['GAME_Luffi2022_CI95_Low_km']    = _game_CIlo
    out['GAME_Luffi2022_CI95_High_km']   = _game_CIhi
    out['GAME_Luffi2022_CI95_Width_km']  = _game_CIw
    out['GAME_Luffi2022_Reliability']    = _game_rel
    out['GAME_Luffi2022_Status']         = _game_stat
    return out

def game_sensor_labels():
    return {sensor: f'{sensor}: {label}' for sensor, label, _ in GAME_SENSORS}

def game_sensor_summary():
    rows = []
    for sensor, label, calib_col in GAME_SENSORS:
        cal = game_calibrators().get(sensor)
        rows.append({
            'Mohometer': sensor,
            'Variable': label,
            'Calibration column': calib_col,
            'Calibration n': int(len(cal['z'])) if cal else 0,
            'LOOCV RMSE elevation [km]': np.round(cal.get('rmse_elev', np.nan), 3) if cal else np.nan,
            'Approx. Moho RMSE [km]': np.round(cal.get('rmse_elev', np.nan) * GAME_ALPHA_DEFAULT, 2) if cal and np.isfinite(cal.get('rmse_elev', np.nan)) else np.nan,
        })
    return tidy_numbers(pd.DataFrame(rows))

def game_sensor_diagnosis(df, rmse_max_elev=2.0):
    """Return a DataFrame explaining why each GAME sensor fired or didn't.

    Columns: Sensor, Variable, Status, N_samples_with_data
    Status values:
      'Not in calibration file'
      'Skipped – rmse_elev {x} > {threshold}'
      'No element data'
      'Fired'
      'Has data but all outside calibration domain'
    """
    calibrators = game_calibrators()
    mgo = pd.to_numeric(df.get('MgO', pd.Series(dtype=float)), errors='coerce') if 'MgO' in df else pd.Series(dtype=float)
    rows = []
    for sensor, label, _ in GAME_SENSORS:
        cal = calibrators.get(sensor)
        if cal is None:
            rows.append({'Sensor': label, 'Status': 'Not in calibration file', 'N_samples_with_data': 0})
            continue
        rmse_elev = cal.get('rmse_elev', np.nan)
        if np.isfinite(rmse_elev) and rmse_elev > rmse_max_elev:
            rows.append({'Sensor': label, 'Status': f'Skipped – rmse_elev {rmse_elev:.2f} > {rmse_max_elev}', 'N_samples_with_data': 0})
            continue
        y = game_series(df, sensor)
        valid = mgo.notna() & y.notna() if len(mgo) == len(y) else pd.Series(False, index=df.index)
        n_valid = int(valid.sum())
        if n_valid == 0:
            rows.append({'Sensor': label, 'Status': 'No element data', 'N_samples_with_data': 0})
            continue
        # Check if any predictions would be in domain
        mgo_vals = mgo[valid].to_numpy(dtype=float)
        y_vals = y[valid].to_numpy(dtype=float)
        raw = np.column_stack([mgo_vals, y_vals])
        scaled = (raw - cal['center']) / cal['scale']
        if cal.get('hull') is not None:
            try:
                in_domain = cal['hull'].find_simplex(scaled) >= 0
            except Exception:
                in_domain = np.ones(len(scaled), dtype=bool)
        else:
            in_domain = np.ones(len(scaled), dtype=bool)
        n_in_domain = int(in_domain.sum())
        if n_in_domain == 0:
            rows.append({'Sensor': label, 'Status': 'Outside calibration domain', 'N_samples_with_data': n_valid})
        else:
            rows.append({'Sensor': label, 'Status': 'Fired', 'N_samples_with_data': n_in_domain})
    return pd.DataFrame(rows)


def _profile_dataset_mgo(df):
    """Summarise the user's MgO distribution.

    Returns dict with median, p5, p95, iqr, frac_with_mgo, and a coarse
    'composition' tag in {'mafic','intermediate','felsic','mixed','unknown'}.
    """
    out = {'median': np.nan, 'p5': np.nan, 'p95': np.nan, 'iqr': np.nan,
           'frac_with_mgo': 0.0, 'composition': 'unknown', 'n': 0}
    if df is None or df.empty or 'MgO' not in df:
        return out
    mgo = pd.to_numeric(df['MgO'], errors='coerce').dropna()
    out['n'] = int(len(mgo))
    if len(mgo) == 0:
        return out
    out['frac_with_mgo'] = float(len(mgo) / max(len(df), 1))
    out['median'] = float(mgo.median())
    out['p5']  = float(mgo.quantile(0.05))
    out['p95'] = float(mgo.quantile(0.95))
    out['iqr'] = float(mgo.quantile(0.75) - mgo.quantile(0.25))
    # Composition tag — IQR threshold of 4 wt% catches strongly bimodal sets.
    if out['iqr'] > 4:
        out['composition'] = 'mixed'
    elif out['median'] >= 6:
        out['composition'] = 'mafic'
    elif out['median'] >= 2:
        out['composition'] = 'intermediate'
    else:
        out['composition'] = 'felsic'
    return out


def _has_required_elements(df, required):
    """Check if all required elements are columns in df. Tuples-of-tuples
    mean any one of the alternatives suffices (e.g. (('FeO','Fe2O3T'),))."""
    if df is None:
        return False
    cols = set(df.columns)
    for req in required:
        if isinstance(req, tuple):
            if not any(alt in cols for alt in req):
                return False
        else:
            if req not in cols:
                return False
    return True


def recommend_sensors(df, mgo_overlap_threshold=0.8, profile=None):
    """Recommend a GAME sensor subset for this dataset, grounded in
    Luffi & Ducea (2022) Table 1 + §5.5.1.

    Returns dict with:
      'recommended'     : list of sensor IDs that pass all gates
      'caution'         : sensors that fire but have weakness in this dataset
      'not_recommended' : sensors that fail element availability or MgO range
      'reasons'         : per-sensor short string explaining the verdict
      'profile'         : output of _profile_dataset_mgo (for display)
    """
    profile = profile or _profile_dataset_mgo(df)
    composition = profile['composition']
    rec, caution, bad = [], [], []
    reasons = {}
    if df is None or df.empty:
        return {'recommended': [], 'caution': [], 'not_recommended': [],
                'reasons': {}, 'profile': profile}

    # MgO overlap gate: what fraction of samples falls within the sensor's
    # calibrated MgO range. Soft-warn below threshold; hard-fail at zero.
    mgo = pd.to_numeric(df['MgO'], errors='coerce').dropna() if 'MgO' in df else pd.Series(dtype=float)
    n_with_mgo = max(len(mgo), 1)

    for sensor, (mgo_lo, mgo_hi, diff_class, r2, required) in GAME_SENSOR_META.items():
        # Gate 1 — required elements present
        if not _has_required_elements(df, required):
            missing = [(r if not isinstance(r, tuple) else '|'.join(r)) for r in required
                       if (isinstance(r, tuple) and not any(a in df.columns for a in r))
                          or (not isinstance(r, tuple) and r not in df.columns)]
            bad.append(sensor)
            reasons[sensor] = f'missing element: {",".join(missing)}'
            continue

        # Gate 2 — MgO range overlap
        if len(mgo) > 0:
            in_range = ((mgo >= mgo_lo) & (mgo <= mgo_hi)).sum()
            overlap = in_range / n_with_mgo
        else:
            overlap = 0.0

        if overlap == 0.0:
            bad.append(sensor)
            reasons[sensor] = (f'MgO range mismatch: calibrated {mgo_lo}–{mgo_hi}, '
                               f'your MgO {profile["p5"]:.1f}–{profile["p95"]:.1f}')
            continue
        if overlap < mgo_overlap_threshold:
            caution.append(sensor)
            reasons[sensor] = (f'only {overlap*100:.0f}% of samples in calibrated '
                               f'MgO {mgo_lo}–{mgo_hi}')
            continue

        # Gate 3 — differentiation class vs dataset composition
        if composition == 'mixed' and diff_class == 'strong':
            caution.append(sensor)
            reasons[sensor] = (f'strongly MgO-sensitive — risky on mixed dataset '
                               f'(IQR {profile["iqr"]:.1f} wt%)')
            continue
        if composition == 'mixed' and diff_class == 'mild':
            caution.append(sensor)
            reasons[sensor] = 'mildly MgO-sensitive on mixed dataset'
            continue

        rec.append(sensor)
        reasons[sensor] = f'OK ({overlap*100:.0f}% in MgO range, R²={r2:.3f}, {diff_class})'

    return {'recommended': rec, 'caution': caution, 'not_recommended': bad,
            'reasons': reasons, 'profile': profile}


def get_game_preset(preset_name, df=None):
    """Resolve a preset name to a list of sensor IDs.
    Returns None for 'Custom' (UI handles its own state)."""
    if preset_name == 'All 41':
        return list(GAME_SENSOR_META.keys())
    if preset_name == 'Top 10 by R²':
        return list(GAME_TOP10_BY_R2)
    if preset_name == 'MgO-insensitive only':
        return list(GAME_MGO_INSENSITIVE)
    if preset_name == 'Mafic (MgO 5–10)':
        # MgO-insensitive (work everywhere) + strongly-MgO-sensitive sensors,
        # which happen to be the ones whose signal is *strongest* in the
        # mafic range (Ni, Co, Cr, Th/Yb, Ba/V — all element-rich in basalts).
        # Filter to sensors whose calibrated MgO band covers ≥8 wt%.
        return [s for s,(lo,hi,cls,_,_) in GAME_SENSOR_META.items()
                if cls in ('insensitive','strong') and lo <= 5 and hi >= 8]
    if preset_name == 'Intermediate (MgO 2–5)':
        # Top R² mohometers all have median data support in this band.
        # Use the paper-endorsed top-10 (§5.5.2) as the default for
        # andesite–dacite compositions.
        return list(GAME_TOP10_BY_R2)
    if preset_name == 'Felsic (MgO < 2)':
        # Exclude strongly-MgO-sensitive sensors — Ni, Co, Cr, Th/Yb, Th/Y,
        # Ba/V, A/CaO need mafic-range data to discriminate. Also reject
        # sensors whose lower MgO bound > 1 (e.g. Co 3–9).
        return [s for s,(lo,hi,cls,_,_) in GAME_SENSOR_META.items()
                if cls != 'strong' and lo <= 1 and hi >= 2]
    if preset_name == 'Auto from data' and df is not None:
        rec = recommend_sensors(df)
        return rec['recommended'] or rec['recommended'] + rec['caution']
    if preset_name == 'Custom':
        return None
    return list(GAME_SENSOR_META.keys())  # safe default = all


def recompute_game_consensus(df, sensor_subset=None,
                             focus_mad_km=5.0, final_mad_km=10.0,
                             min_mohometers=3, seed=42):
    """Recompute the GAME consensus median/MAD/CI from already-stored
    per-sensor `GAME_{sensor}_H_km` columns, restricted to `sensor_subset`.

    Returns the input df with consensus columns overwritten. Cheap — does
    not rerun any LOWESS surfaces.
    """
    out = df.copy()
    all_sensors = list(GAME_SENSOR_META.keys())
    use = list(sensor_subset) if sensor_subset is not None else all_sensors
    cols = [f'GAME_{s}_H_km' for s in use if f'GAME_{s}_H_km' in out.columns]
    if not cols:
        out['H_GAME_LuffiDucea2022_km'] = np.nan
        out['GAME_Luffi2022_N_mohometers'] = 0
        out['GAME_Luffi2022_N_raw_mohometers'] = 0
        out['GAME_Luffi2022_MAD_km'] = np.nan
        out['GAME_Luffi2022_IQR_km'] = np.nan
        out['GAME_Luffi2022_CI95_Low_km'] = np.nan
        out['GAME_Luffi2022_CI95_High_km'] = np.nan
        out['GAME_Luffi2022_CI95_Width_km'] = np.nan
        out['GAME_Luffi2022_Reliability'] = 'No estimate'
        out['GAME_Luffi2022_Status'] = 'no sensors selected'
        return out

    est_arr = out[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    n_rows = len(out)
    H   = np.full(n_rows, np.nan)
    N   = np.zeros(n_rows, int)
    Nr  = np.zeros(n_rows, int)
    MAD = np.full(n_rows, np.nan)
    IQR = np.full(n_rows, np.nan)
    CIl = np.full(n_rows, np.nan)
    CIh = np.full(n_rows, np.nan)
    CIw = np.full(n_rows, np.nan)
    rel = np.full(n_rows, 'No estimate', dtype=object)
    stat= np.full(n_rows, 'no usable mohometers', dtype=object)
    for pos in range(n_rows):
        row_vals = est_arr[pos]
        vals = row_vals[np.isfinite(row_vals)]
        Nr[pos] = len(vals)
        if len(vals) < min_mohometers:
            N[pos] = len(vals)
            stat[pos] = f'low N ({len(vals)} mohometers)'
            rel[pos] = 'Low N'
            continue
        raw_median = float(np.nanmedian(vals))
        dev = np.abs(vals - raw_median)
        keep = dev <= focus_mad_km
        status = 'focus MAD'
        if int(keep.sum()) < min_mohometers:
            keep = dev <= final_mad_km
            status = 'final MAD'
        if int(keep.sum()) < min_mohometers:
            keep = np.isfinite(vals)
            status = 'all valid; high spread'
        kept = vals[keep]
        if len(kept) < min_mohometers:
            N[pos] = len(kept)
            stat[pos] = 'low N after MAD filtering'
            rel[pos] = 'Low N'
            continue
        q25, q75 = np.nanpercentile(kept, [25, 75])
        ci_low, ci_high = _bootstrap_ci_median(kept, seed=seed + pos)
        mad = float(np.nanmedian(np.abs(kept - np.nanmedian(kept))))
        ci_w = float(ci_high - ci_low) if np.isfinite(ci_low) and np.isfinite(ci_high) else np.nan
        if len(kept) < 8 or mad > 10 or (np.isfinite(ci_w) and ci_w > 15) or status == 'all valid; high spread':
            reliability = 'Low confidence'
        elif len(kept) < 15 or mad > 5 or (np.isfinite(ci_w) and ci_w > 8):
            reliability = 'Caution'
        else:
            reliability = 'Good'
        H[pos]  = float(np.nanmedian(kept))
        N[pos]  = len(kept)
        MAD[pos]= mad
        IQR[pos]= float(q75 - q25)
        CIl[pos]= ci_low
        CIh[pos]= ci_high
        CIw[pos]= ci_w
        rel[pos]= reliability
        stat[pos]= f'{status}; {len(kept)} of {len(vals)} mohometers'
    out['H_GAME_LuffiDucea2022_km']      = H
    out['GAME_Luffi2022_N_mohometers']   = N
    out['GAME_Luffi2022_N_raw_mohometers'] = Nr
    out['GAME_Luffi2022_MAD_km']         = MAD
    out['GAME_Luffi2022_IQR_km']         = IQR
    out['GAME_Luffi2022_CI95_Low_km']    = CIl
    out['GAME_Luffi2022_CI95_High_km']   = CIh
    out['GAME_Luffi2022_CI95_Width_km']  = CIw
    out['GAME_Luffi2022_Reliability']    = rel
    out['GAME_Luffi2022_Status']         = stat
    return out


# Sensors plotted on log y-axis in Luffi & Ducea (2022) Figure S1 —
# all element ratios and trace-element ppm sensors. Major-oxide wt%
# sensors (MnO, K2O, CaO, FeOt, A) stay linear.
GAME_LOG_Y_SENSORS = {
    # Element ratios (values can span 1–500 in arc rocks)
    'LaYb','CeYb','NdYb','LaY','LuHf','SmYb','NdY','NbY','CeY','LaSm',
    'ThYb','GdYb','ZrY','BaV','BaSc','NbYb','ThY','DyYb','NiSc','CrSc',
    'NiV','CrV','ZrTi','ACaO','SrYx',
    # Trace elements in ppm (orders of magnitude)
    'Hf','Sc','Pb','Rb','Ba','Ni','Sr','U','Ga','Co','Cr',
}


def game_calibration_figure(sample_df, sensor):
    cal = game_calibrators().get(sensor)
    if cal is None:
        return None
    labels = game_sensor_labels()
    raw_xy = cal['raw_xy']
    use_log_y = sensor in GAME_LOG_Y_SENSORS

    x_grid = np.linspace(float(np.nanmin(raw_xy[:,0])), float(np.nanmax(raw_xy[:,0])), 34)
    y_min = float(np.nanmin(raw_xy[:,1]))
    y_max = float(np.nanmax(raw_xy[:,1]))
    # Log-spaced grid on log axes so contour cells are uniform after rendering.
    # Fall back to linear if any non-positive values would break log10.
    if use_log_y and y_min > 0 and y_max > 0:
        y_grid = np.logspace(np.log10(y_min), np.log10(y_max), 34)
    else:
        y_grid = np.linspace(y_min, y_max, 34)
        use_log_y = False
    z_grid = np.full((len(y_grid), len(x_grid)), np.nan)
    for yi, yv in enumerate(y_grid):
        pts = np.column_stack([x_grid, np.full(len(x_grid), yv)])
        scaled = (pts - cal['center']) / cal['scale']
        if cal.get('hull') is not None:
            try:
                in_domain = cal['hull'].find_simplex(scaled) >= 0
            except Exception:
                in_domain = np.ones(len(scaled), dtype=bool)
        else:
            in_domain = np.ones(len(scaled), dtype=bool)
        for xi, (pt, ok) in enumerate(zip(scaled, in_domain)):
            if ok:
                z_grid[yi,xi] = _game_loess_predict_one(cal, pt)
    fig = go.Figure()
    moho_grid = GAME_ALPHA_DEFAULT * z_grid + GAME_BETA_DEFAULT
    fig.add_trace(go.Contour(
        x=x_grid, y=y_grid, z=moho_grid,
        colorscale='Viridis', contours=dict(showlabels=True, labelfont=dict(size=9)),
        colorbar=dict(title='Moho [km]'), name='LOWESS-style surface',
        hovertemplate='MgO=%{x:.1f}<br>'+labels.get(sensor,sensor)+'=%{y:.2f}<br>surface Moho=%{z:.1f} km<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=raw_xy[:,0], y=raw_xy[:,1], mode='markers', name='T2 calibration points',
        marker=dict(size=6,color='rgba(255,255,255,0.85)',line=dict(color='black',width=0.7)),
        customdata=np.c_[cal['z'], GAME_ALPHA_DEFAULT * cal['z'] + GAME_BETA_DEFAULT],
        hovertemplate='MgO=%{x:.1f}<br>'+labels.get(sensor,sensor)+'=%{y:.2f}<br>calibration elevation=%{customdata[0]:.1f} km<br>calibration Moho=%{customdata[1]:.1f} km<extra></extra>'
    ))
    if sample_df is not None and not sample_df.empty and 'MgO' in sample_df:
        y = game_series(sample_df, sensor)
        overlay = pd.DataFrame({'MgO':pd.to_numeric(sample_df['MgO'], errors='coerce'), 'Mohometer':pd.to_numeric(y, errors='coerce')}).dropna()
        # Drop non-positive y for log-axis to avoid plotly warnings
        if use_log_y:
            overlay = overlay[overlay['Mohometer'] > 0]
        if not overlay.empty:
            fig.add_trace(go.Scatter(
                x=overlay['MgO'], y=overlay['Mohometer'], mode='markers', name='uploaded samples',
                marker=dict(size=3.5,color='rgba(60,60,60,0.30)',symbol='circle',line=dict(width=0)),
                hovertemplate='sample MgO=%{x:.1f}<br>'+labels.get(sensor,sensor)+'=%{y:.2f}<extra></extra>'
            ))
    fig.update_layout(template='plotly_white',height=460,margin=dict(l=20,r=20,t=20,b=20),legend=dict(orientation='h',y=-0.22,x=0))
    fig.update_xaxes(title='MgO [wt%]',showgrid=True)
    _y_title = labels.get(sensor, sensor) + (' (log scale)' if use_log_y else '')
    fig.update_yaxes(title=_y_title, showgrid=True,
                     type='log' if use_log_y else 'linear')
    return fig

def game_consensus_figure(df, x_col='Observed_km', point_size=5):
    if df.empty or 'H_GAME_LuffiDucea2022_km' not in df or x_col not in df:
        return None
    plot = df.copy()
    plot[x_col] = pd.to_numeric(plot[x_col], errors='coerce')
    plot['H_GAME_LuffiDucea2022_km'] = pd.to_numeric(plot['H_GAME_LuffiDucea2022_km'], errors='coerce')
    plot = plot.dropna(subset=[x_col,'H_GAME_LuffiDucea2022_km'])
    if plot.empty:
        return None
    plot['GAME_delta_km'] = plot['H_GAME_LuffiDucea2022_km'] - plot[x_col]
    for c, default in {
        'GAME_Luffi2022_N_mohometers':0,
        'GAME_Luffi2022_N_raw_mohometers':0,
        'GAME_Luffi2022_MAD_km':np.nan,
        'GAME_Luffi2022_IQR_km':np.nan,
        'GAME_Luffi2022_CI95_Width_km':np.nan,
        'GAME_Luffi2022_Reliability':'',
        'GAME_Luffi2022_Status':'',
    }.items():
        if c not in plot:
            plot[c] = default
    err_plus = None
    err_minus = None
    if {'GAME_Luffi2022_CI95_Low_km','GAME_Luffi2022_CI95_High_km'}.issubset(plot):
        err_plus = (pd.to_numeric(plot['GAME_Luffi2022_CI95_High_km'], errors='coerce') - plot['H_GAME_LuffiDucea2022_km']).clip(lower=0)
        err_minus = (plot['H_GAME_LuffiDucea2022_km'] - pd.to_numeric(plot['GAME_Luffi2022_CI95_Low_km'], errors='coerce')).clip(lower=0)
    x_label = 'known: crustal thickness [Km]' if x_col == 'Observed_km' else local_option_label(x_col)
    fig = px.scatter(
        tidy_numbers(plot), x=x_col, y='H_GAME_LuffiDucea2022_km', color='GAME_delta_km',
        color_continuous_scale=RESIDUAL_COLORSCALE, color_continuous_midpoint=0, template='plotly_white',
        labels={x_col:x_label,'H_GAME_LuffiDucea2022_km':'Luffi & Ducea GAME: crustal thickness [Km]','GAME_delta_km':'GAME - reference [Km]','GAME_Luffi2022_N_mohometers':'mohometers'},
        custom_data=[c for c in ['GAME_delta_km','GAME_Luffi2022_N_mohometers','GAME_Luffi2022_N_raw_mohometers','GAME_Luffi2022_MAD_km','GAME_Luffi2022_IQR_km','GAME_Luffi2022_CI95_Width_km','GAME_Luffi2022_Reliability','GAME_Luffi2022_Status'] if c in plot]
    )
    fig.update_traces(marker=dict(size=point_size,line=dict(color='black',width=0.6)),hovertemplate=x_label+'=%{x:.1f}<br>GAME=%{y:.1f} Km<br>Delta=%{customdata[0]:+.1f} Km<br>N kept=%{customdata[1]:.0f}<br>N raw=%{customdata[2]:.0f}<br>MAD=%{customdata[3]:.1f} Km<br>IQR=%{customdata[4]:.1f} Km<br>CI width=%{customdata[5]:.1f} Km<br>%{customdata[6]}<br>%{customdata[7]}<extra></extra>')
    if err_plus is not None and err_minus is not None:
        fig.update_traces(error_y=dict(type='data',array=err_plus.to_numpy(),arrayminus=err_minus.to_numpy(),visible=True,thickness=0.7,width=2,color='rgba(70,70,70,0.45)'))
    fig.add_trace(go.Scatter(x=[0,90],y=[0,90],mode='lines',name='1:1',line=dict(color='black',dash='dash')))
    fig.update_layout(height=460,margin=dict(l=20,r=20,t=20,b=20),legend=dict(orientation='h',y=-0.22,x=0))
    fig.update_xaxes(range=[0,90],showgrid=True)
    fig.update_yaxes(range=[0,90],showgrid=True)
    return fig

def game_reliability_figure(df, x_col='Observed_km', point_size=6):
    required = ['GAME_Luffi2022_N_mohometers','H_GAME_LuffiDucea2022_km']
    if df is None or df.empty or not all(c in df for c in required):
        return None
    plot = df.copy()
    plot['GAME_Luffi2022_N_mohometers'] = pd.to_numeric(plot['GAME_Luffi2022_N_mohometers'], errors='coerce')
    plot['H_GAME_LuffiDucea2022_km'] = pd.to_numeric(plot['H_GAME_LuffiDucea2022_km'], errors='coerce')
    for c in ['GAME_Luffi2022_MAD_km','GAME_Luffi2022_IQR_km','GAME_Luffi2022_CI95_Width_km','GAME_Luffi2022_N_raw_mohometers']:
        if c in plot:
            plot[c] = pd.to_numeric(plot[c], errors='coerce')
        else:
            plot[c] = np.nan
    if 'GAME_Luffi2022_Reliability' not in plot:
        plot['GAME_Luffi2022_Reliability'] = ''
    if 'GAME_Luffi2022_Status' not in plot:
        plot['GAME_Luffi2022_Status'] = ''
    has_reference = x_col in plot and pd.to_numeric(plot[x_col], errors='coerce').notna().any()
    if has_reference:
        plot[x_col] = pd.to_numeric(plot[x_col], errors='coerce')
        plot['GAME_delta_km'] = plot['H_GAME_LuffiDucea2022_km'] - plot[x_col]
        plot['GAME_abs_delta_km'] = plot['GAME_delta_km'].abs()
        y_col = 'GAME_abs_delta_km'
        y_title = '|GAME - reference| [Km]'
        color_col = 'GAME_delta_km'
        color_title = 'GAME - reference [Km]'
    else:
        y_col = 'GAME_Luffi2022_MAD_km'
        y_title = 'GAME MAD [Km]'
        color_col = 'GAME_Luffi2022_MAD_km'
        color_title = 'GAME MAD [Km]'
    plot = plot.dropna(subset=['GAME_Luffi2022_N_mohometers', y_col])
    if plot.empty:
        return None
    fig = go.Figure()
    marker = dict(
        size=point_size,
        color=plot[color_col],
        colorscale=RESIDUAL_COLORSCALE if color_col == 'GAME_delta_km' else 'Viridis',
        colorbar=dict(title=color_title),
        line=dict(color='black', width=0.6),
        opacity=0.86,
    )
    if color_col == 'GAME_delta_km':
        marker['cmid'] = 0
    delta_series = plot['GAME_delta_km'] if 'GAME_delta_km' in plot else pd.Series(np.nan, index=plot.index)
    fig.add_trace(go.Scatter(
        x=plot['GAME_Luffi2022_N_mohometers'], y=plot[y_col], mode='markers',
        name='GAME samples', marker=marker,
        customdata=np.c_[plot['GAME_Luffi2022_N_raw_mohometers'], plot['GAME_Luffi2022_MAD_km'], plot['GAME_Luffi2022_IQR_km'], plot['GAME_Luffi2022_CI95_Width_km'], delta_series, plot['GAME_Luffi2022_Reliability'], plot['GAME_Luffi2022_Status']],
        hovertemplate='N kept=%{x:.0f}<br>'+y_title+'=%{y:.1f}<br>N raw=%{customdata[0]:.0f}<br>MAD=%{customdata[1]:.1f} Km<br>IQR=%{customdata[2]:.1f} Km<br>CI width=%{customdata[3]:.1f} Km<br>Delta=%{customdata[4]:+.1f} Km<br>%{customdata[5]}<br>%{customdata[6]}<extra></extra>'
    ))
    fig.add_vline(x=3, line=dict(color='rgba(17,24,39,0.45)', dash='dash', width=1), annotation_text='minimum in app', annotation_position='top left')
    fig.add_vline(x=10, line=dict(color='rgba(17,24,39,0.28)', dash='dot', width=1), annotation_text='more stable', annotation_position='top right')
    fig.update_layout(template='plotly_white',height=380,margin=dict(l=20,r=20,t=20,b=20),xaxis_title='GAME mohometers kept after filtering',yaxis_title=y_title,showlegend=False)
    fig.update_xaxes(dtick=5, rangemode='tozero')
    fig.update_yaxes(rangemode='tozero')
    return fig

def game_reliability_summary(df, x_col='Observed_km'):
    if df is None or df.empty or 'GAME_Luffi2022_N_mohometers' not in df:
        return pd.DataFrame()
    data = df.copy()
    data['GAME_Luffi2022_N_mohometers'] = pd.to_numeric(data['GAME_Luffi2022_N_mohometers'], errors='coerce')
    for c in ['GAME_Luffi2022_MAD_km','GAME_Luffi2022_IQR_km','GAME_Luffi2022_CI95_Width_km','H_GAME_LuffiDucea2022_km']:
        if c in data:
            data[c] = pd.to_numeric(data[c], errors='coerce')
    if x_col in data and 'H_GAME_LuffiDucea2022_km' in data:
        data[x_col] = pd.to_numeric(data[x_col], errors='coerce')
        data['Abs_Delta_km'] = (data['H_GAME_LuffiDucea2022_km'] - data[x_col]).abs()
    group = data.dropna(subset=['GAME_Luffi2022_N_mohometers']).groupby('GAME_Luffi2022_N_mohometers', dropna=False)
    rows = []
    for n, g in group:
        row = {
            'N_mohometers': int(n) if pd.notna(n) else np.nan,
            'Rows': len(g),
            'Median_MAD_km': pd.to_numeric(g.get('GAME_Luffi2022_MAD_km', pd.Series(dtype=float)), errors='coerce').median(),
            'Median_IQR_km': pd.to_numeric(g.get('GAME_Luffi2022_IQR_km', pd.Series(dtype=float)), errors='coerce').median(),
            'Median_CI95_Width_km': pd.to_numeric(g.get('GAME_Luffi2022_CI95_Width_km', pd.Series(dtype=float)), errors='coerce').median(),
        }
        if 'Abs_Delta_km' in g:
            row['Median_Abs_Delta_km'] = pd.to_numeric(g['Abs_Delta_km'], errors='coerce').median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values('N_mohometers')

@st.cache_data(show_spinner=False)
def enrich(df,la_yb_mode='raw_ppm',include_game=False):
    out = df.copy(); sio2 = out['SiO2'] if 'SiO2' in out else pd.Series(np.nan,index=out.index); mgo = out['MgO'] if 'MgO' in out else pd.Series(np.nan,index=out.index)
    text_class = out['Rock_Type'].map(classify_rock_text) if 'Rock_Type' in out else pd.Series(np.nan,index=out.index)
    out['Rock_Type_Text_Class'] = text_class
    out['Rock_Type_Model'] = text_class.fillna('unclassified')
    out.loc[out['Rock_Type_Model'].eq('unclassified') & (sio2>=44)&(sio2<=53)&(mgo>4),'Rock_Type_Model']='mafic'
    out.loc[out['Rock_Type_Model'].eq('unclassified') & (sio2>=55)&(sio2<=68),'Rock_Type_Model']='intermediate'
    out.loc[out['Rock_Type_Model'].eq('unclassified') & (sio2>68),'Rock_Type_Model']='felsic'
    if {'Sr','Y'}.issubset(out): out['Sr_Y']=div(out['Sr'],out['Y'])
    if {'La','Yb'}.issubset(out):
        out['La_Yb_raw']=div(out['La'],out['Yb']); out['La_Yb_N']=out['La_Yb_raw'] if la_yb_mode=='already_normalized' else out['La_Yb_raw']/(CHON['La']/CHON['Yb'])
    elif 'La_Yb_raw' in out and 'La_Yb_N' not in out:
        out['La_Yb_N']=out['La_Yb_raw'] if la_yb_mode=='already_normalized' else out['La_Yb_raw']/(CHON['La']/CHON['Yb'])
    elif 'La_Yb_N' in out and 'La_Yb_raw' not in out and la_yb_mode=='already_normalized':
        out['La_Yb_raw']=out['La_Yb_N']
    if {'Ce','Y'}.issubset(out): out['Ce_Y']=div(out['Ce'],out['Y'])
    if {'La','Y'}.issubset(out): out['La_Y']=div(out['La'],out['Y'])
    if {'MnO','MgO'}.issubset(out): out['MnO_MgO']=div(out['MnO'],out['MgO'])
    if {'Dy','Yb'}.issubset(out): out['Dy_Yb']=div(out['Dy'],out['Yb'])
    if {'Gd','Yb'}.issubset(out): out['Gd_Yb']=div(out['Gd'],out['Yb'])
    if {'Sm','Eu','Gd'}.issubset(out): out['Eu_Eu_star']=(out['Eu']/CHON['Eu'])/np.sqrt((out['Sm']/CHON['Sm'])*(out['Gd']/CHON['Gd']))
    if {'Rb','Sr'}.issubset(out): out['Rb_Sr']=div(out['Rb'],out['Sr'])
    if 'Sr_Y' in out: out['H_Profeta2015_SrY_km']=(out['Sr_Y']+7.25)/0.90; out['H_Sundell2021_SrY_km']=19.6*pln(out['Sr_Y'])-24; out['H_Zou2021_SrY_SVRE_km']=1.11*out['Sr_Y']+8.05
    if 'Sr_Y' in out: out['H_Hu2017_Collisional_SrY_km']=0.67*out['Sr_Y']+28.21
    if 'La_Yb_N' in out: out['H_Profeta2015_LaYbN_km']=np.where(out['La_Yb_N']>0,np.log(out['La_Yb_N']/0.98)/0.047,np.nan); out['H_Sundell2021_LaYbN_km']=17*pln(out['La_Yb_N'])+6.9; out['H_Zou2021_LaYbN_SVRE_km']=21.277*np.log(1.0204*out['La_Yb_N'])
    if 'La_Yb_N' in out: out['H_Hu2017_Collisional_LaYbN_km']=np.where(0.34*out['La_Yb_N']>0,27.78*np.log(0.34*out['La_Yb_N']),np.nan)
    if {'Sr_Y','La_Yb_N'}.issubset(out): out['H_Sundell2021_Paired_km']=10.3*pln(out['Sr_Y'])+8.8*pln(out['La_Yb_N'])-10.6
    if 'Ce_Y' in out: out['H_Mantle2008_CeY_sample_km']=np.where(out['Ce_Y']>0,np.log(out['Ce_Y']/0.3029)/0.0554,np.nan)
    if 'Rb_Sr' in out: out['H_Dhuime2015_RbSr_km']=426.8*out['Rb_Sr']+4.1
    if 'SiO2' in out:
        out['H_Dhuime2015_SiO2_km']=3.5*pd.to_numeric(out['SiO2'],errors='coerce')-157.8
        out['Elevation_FarnerLee2017_SiO2_km']=0.81*pd.to_numeric(out['SiO2'],errors='coerce')-45.82
        out['H_FarnerLee2017_SiO2_ElevMoho_km']=GAME_ALPHA_DEFAULT*out['Elevation_FarnerLee2017_SiO2_km']+GAME_BETA_DEFAULT
    if 'FeO' in out:
        out['Elevation_FarnerLee2017_FeOt_km']=-1.70*pd.to_numeric(out['FeO'],errors='coerce')+12.78
        out['H_FarnerLee2017_FeOt_ElevMoho_km']=GAME_ALPHA_DEFAULT*out['Elevation_FarnerLee2017_FeOt_km']+GAME_BETA_DEFAULT
    if 'CaO' in out:
        out['Elevation_FarnerLee2017_CaO_km']=-1.57*pd.to_numeric(out['CaO'],errors='coerce')+12.18
        out['H_FarnerLee2017_CaO_ElevMoho_km']=GAME_ALPHA_DEFAULT*out['Elevation_FarnerLee2017_CaO_km']+GAME_BETA_DEFAULT
    if 'K2O' in out:
        out['Elevation_FarnerLee2017_K2O_km']=3.27*pd.to_numeric(out['K2O'],errors='coerce')-4.54
        out['H_FarnerLee2017_K2O_ElevMoho_km']=GAME_ALPHA_DEFAULT*out['Elevation_FarnerLee2017_K2O_km']+GAME_BETA_DEFAULT
    if 'Elevation_km' in out:
        out['H_LuffiDucea2022_Elevation_Moho_km']=GAME_ALPHA_DEFAULT*pd.to_numeric(out['Elevation_km'],errors='coerce')+GAME_BETA_DEFAULT
    if include_game:
        out = apply_luffi_game_mohometers(out)
    out['Preferred_H_km']=out['H_Guo_ERT_km'] if 'H_Guo_ERT_km' in out else np.nan; out['Preferred_Method']=np.where(pd.notna(out['Preferred_H_km']),'Guo ERT ML','none')
    if 'H_Sundell2021_Paired_km' in out:
        m = pd.isna(out['Preferred_H_km']); out.loc[m,'Preferred_H_km']=out.loc[m,'H_Sundell2021_Paired_km']; out.loc[m,'Preferred_Method']='Sundell paired'
    if 'H_GAME_LuffiDucea2022_km' in out:
        m = pd.isna(out['Preferred_H_km']); out.loc[m,'Preferred_H_km']=out.loc[m,'H_GAME_LuffiDucea2022_km']; out.loc[m,'Preferred_Method']='Luffi & Ducea GAME'
    _mafic = (out['Rock_Type_Model'].to_numpy() == 'mafic') if 'Rock_Type_Model' in out else np.zeros(len(out), bool)
    _loi = (pd.to_numeric(out['LOI'], errors='coerce') > 3).fillna(False).to_numpy() if 'LOI' in out else np.zeros(len(out), bool)
    if 'Sr_Y' in out and 'Sr' in out:
        _sry = ((pd.to_numeric(out['Sr_Y'], errors='coerce') > 80) & (pd.to_numeric(out['Sr'], errors='coerce') > 1000)).fillna(False).to_numpy()
    else:
        _sry = np.zeros(len(out), bool)
    if 'GAME_Luffi2022_Reliability' in out:
        _game_rel_lower = out['GAME_Luffi2022_Reliability'].fillna('').astype(str).str.lower().to_numpy()
        _game_flag = np.isin(_game_rel_lower, ['caution', 'low confidence', 'low n'])
        _game_rel_vals = out['GAME_Luffi2022_Reliability'].fillna('').astype(str).to_numpy()
        _game_n_vals = out.get('GAME_Luffi2022_N_mohometers', pd.Series(0, index=out.index)).fillna(0).astype(int).astype(str).to_numpy()
    else:
        _game_flag = np.zeros(len(out), bool)
        _game_rel_vals = _game_n_vals = np.full(len(out), '')
    flags = []
    for i in range(len(out)):
        f = []
        if _mafic[i]: f.append('Mafic: intermediate Sr/Y-La/Yb proxies caution')
        if _loi[i]: f.append('High LOI alteration caution')
        if _sry[i]: f.append('High Sr/Y + high Sr: possible cumulate/high-Sr effect')
        if _game_flag[i]: f.append(f'GAME {_game_rel_vals[i]}: {_game_n_vals[i]} mohometers')
        flags.append('; '.join(f) if f else 'OK')
    out['Reliability_Flags'] = flags
    return out

def curves(proxy):
    H=np.linspace(5,90,300); d={}
    if proxy=='Sr_Y': d={'Profeta 2015':0.90*H-7.25,'Sundell 2021':np.exp((H+24)/19.6),'Zou 2021':(H-8.05)/1.11,'Hu 2017 collisional':(H-28.21)/0.67}
    if proxy=='La_Yb_N': d={'Profeta 2015':0.98*np.exp(0.047*H),'Sundell 2021':np.exp((H-6.9)/17),'Zou 2021':np.exp(H/21.277)/1.0204,'Hu 2017 collisional':np.exp(H/27.78)/0.34}
    if proxy=='Ce_Y': d={'Mantle & Collins 2008':0.3029*np.exp(0.0554*H),'M&C +3 km':0.3029*np.exp(0.0554*(H+3)),'M&C -3 km':0.3029*np.exp(0.0554*(H-3))}
    if proxy=='Rb_Sr': d={'Dhuime 2015':(H-4.1)/426.8}
    if proxy=='SiO2': d={'Dhuime 2015':(H+157.8)/3.5,'Farner & Lee 2017 via elevation':((H-GAME_BETA_DEFAULT)/GAME_ALPHA_DEFAULT+45.82)/0.81}
    if proxy=='Elevation_km': d={'Luffi & Ducea 2022 elevation-Moho':(H-GAME_BETA_DEFAULT)/GAME_ALPHA_DEFAULT}
    return H,d

def proxy_formula_labels(proxy):
    formulas = {
        'Sr_Y': {
            'Profeta 2015':'Sr/Y = 0.90H - 7.25',
            'Sundell 2021':'H = 19.6 ln(Sr/Y) - 24',
            'Zou 2021':'H = 1.11(Sr/Y) + 8.05',
            'Hu 2017 collisional':'H = 0.67(Sr/Y) + 28.21',
        },
        'La_Yb_N': {
            'Profeta 2015':'La/Yb(N) = 0.98 e^(0.047H)',
            'Sundell 2021':'H = 17 ln(La/Yb(N)) + 6.9',
            'Zou 2021':'H = 21.277 ln(1.0204 La/Yb(N))',
            'Hu 2017 collisional':'H = 27.78 ln(0.34 La/Yb(N))',
        },
        'Ce_Y': {
            'Mantle & Collins 2008':'H = 18.0505 ln(Ce/Y) + 21.5587',
            'M&C +3 km':'Mantle & Collins +3 km',
            'M&C -3 km':'Mantle & Collins -3 km',
        },
        'Rb_Sr': {
            'Dhuime 2015':'H = 426.8(Rb/Sr) + 4.1',
        },
        'SiO2': {
            'Dhuime 2015':'H = 3.5 SiO2 - 157.8',
            'Farner & Lee 2017 via elevation':'h = 0.81 SiO2 - 45.82; H = 6.79h + 26.40',
        },
        'Elevation_km': {
            'Luffi & Ducea 2022 elevation-Moho':'H = 6.79 elevation + 26.40',
        },
    }
    return formulas.get(proxy, {})

def add_formula_box(fig, formulas):
    if not formulas:
        return fig
    lines = [f'<b>{name}</b>: {formula}' for name, formula in formulas.items()]
    fig.add_annotation(
        x=0.015, y=0.985, xref='paper', yref='paper',
        text='<br>'.join(lines), showarrow=False, align='left',
        xanchor='left', yanchor='top',
        font=dict(size=10,color='rgba(17,24,39,0.92)'),
        bgcolor='rgba(255,255,255,0.86)', bordercolor='rgba(148,163,184,0.65)',
        borderwidth=1, borderpad=6
    )
    return fig

def binned_summary_frame(df, x_col, y_col, stat='Median', bin_width=5.0, min_n=5, group_col=None):
    if df is None or df.empty or x_col not in df or y_col not in df or stat == 'Off':
        return pd.DataFrame()
    cols = [x_col, y_col] + ([group_col] if group_col and group_col in df else [])
    data = ensure_unique_columns(df[cols].copy())
    data[x_col] = pd.to_numeric(data[x_col], errors='coerce')
    data[y_col] = pd.to_numeric(data[y_col], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    if data.empty:
        return pd.DataFrame()
    bin_width = max(float(bin_width or 5.0), 0.1)
    min_n = max(int(min_n or 1), 1)
    xmin = np.floor(data[x_col].min() / bin_width) * bin_width
    xmax = np.ceil(data[x_col].max() / bin_width) * bin_width
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return pd.DataFrame()
    bins = np.arange(xmin, xmax + bin_width, bin_width)
    if len(bins) < 2:
        return pd.DataFrame()
    data['_bin'] = pd.cut(data[x_col], bins=bins, include_lowest=True)
    group_cols = [group_col, '_bin'] if group_col and group_col in data else ['_bin']
    rows = []
    for key, g in data.groupby(group_cols, dropna=False, observed=False):
        if len(g) < min_n:
            continue
        interval = key[-1] if isinstance(key, tuple) else key
        if not hasattr(interval, 'mid'):
            continue
        x_vals = pd.to_numeric(g[x_col], errors='coerce').dropna()
        y_vals = pd.to_numeric(g[y_col], errors='coerce').dropna()
        if x_vals.empty or y_vals.empty:
            continue
        for one_stat in (['Median', 'Mean'] if stat == 'Both' else [stat]):
            y_value = y_vals.mean() if one_stat == 'Mean' else y_vals.median()
            row = {
                'X': float(interval.mid),
                'Y': float(y_value),
                'n': int(len(g)),
                'Statistic': one_stat,
                'X_Median': float(x_vals.median()),
                'Y_Q25': float(y_vals.quantile(0.25)),
                'Y_Q75': float(y_vals.quantile(0.75)),
            }
            if group_col and group_col in data:
                row[group_col] = key[0] if isinstance(key, tuple) else ''
            rows.append(row)
    return pd.DataFrame(rows)

def add_binned_summary_traces(fig, df, x_col, y_col, stat='Median', bin_width=5.0, min_n=5, group_col=None, name_prefix='moving'):
    summary = binned_summary_frame(df, x_col, y_col, stat, bin_width, min_n, group_col)
    if summary.empty:
        return fig
    line_styles = {
        'Median': dict(color=SUMMARY_MEDIAN_COLOR, width=2.2, dash='dot'),
        'Mean': dict(color=SUMMARY_MEAN_COLOR, width=2.0, dash='dash'),
    }
    group_values = [None]
    if group_col and group_col in summary:
        group_values = list(summary[group_col].dropna().astype(str).unique())
    for group_value in group_values:
        gsum = summary if group_value is None else summary[summary[group_col].astype(str).eq(group_value)]
        for stat_name, g in gsum.groupby('Statistic', dropna=False):
            if g.empty:
                continue
            label_bits = [name_prefix, str(stat_name).lower()]
            if group_value is not None:
                label_bits.append(str(group_value))
            fig.add_trace(go.Scatter(
                x=g['X'], y=g['Y'], mode='lines+markers',
                name=': '.join(label_bits),
                line=line_styles.get(str(stat_name), line_styles['Median']),
                marker=dict(size=7, color=line_styles.get(str(stat_name), line_styles['Median'])['color'], line=dict(color='white', width=0.8)),
                customdata=np.c_[g['n'], g['X_Median'], g['Y_Q25'], g['Y_Q75']],
                hovertemplate='bin centre=%{x:.1f}<br>summary=%{y:.1f}<br>n=%{customdata[0]:.0f}<br>x median=%{customdata[1]:.1f}<br>Q25-Q75=%{customdata[2]:.1f}-%{customdata[3]:.1f}<extra></extra>'
            ))
    return fig

def validation_proxy_figure(df, proxy, x_col, color_by, point_size=5, show_formulas=False, trend_stat='Median', trend_bin_width=5.0, trend_min_n=5, x_range=None, y_range=None):
    proxy_labels = {c:proxy_value_label(c) for c in PROXY_VALUE_LIBRARY}
    model_suffix = ''
    if x_col == 'Predicted_km' and 'Model' in df and df['Model'].nunique(dropna=True) == 1:
        model_suffix = f' ({df["Model"].dropna().astype(str).iloc[0]})'
    x_labels = {'Observed_km':'known: crustal thickness [Km]','Predicted_km':f'model: crustal thickness{model_suffix} [Km]'}
    proxy_label = proxy_labels.get(proxy, proxy.replace('_','/'))
    x_label = x_labels.get(x_col, PROXY_THICKNESS_LABELS.get(x_col, x_col.replace('_',' ')) + (' [Km]' if x_col in PROXY_THICKNESS_LABELS else ''))
    plot = plot_df(df)
    if proxy not in plot or x_col not in plot:
        return None, {}
    dedupe = [c for c in ['Sample_ID','Lat','Lon',x_col,proxy] if c in plot]
    if x_col == 'Predicted_km' and 'Model' in plot:
        dedupe.append('Model')
    plot = plot.drop_duplicates(subset=dedupe if dedupe else None)
    plot = plot.dropna(subset=[x_col, proxy])
    if plot.empty:
        return None, {}
    H, curve_set = curves(proxy)
    formula_lookup = proxy_formula_labels(proxy)
    fig = go.Figure()
    curve_colors = ['#1f77b4','#2ca02c','#9467bd','#ff7f0e']
    for i, (name, values) in enumerate(curve_set.items()):
        values = np.asarray(values, dtype=float)
        ok = np.isfinite(H) & np.isfinite(values) & (values > 0)
        if ok.any():
            formula = formula_lookup.get(name, '')
            fig.add_trace(go.Scatter(
                x=H[ok], y=values[ok], mode='lines', name=name,
                line=dict(color=curve_colors[i % len(curve_colors)], width=1.8),
                hovertemplate='Thickness=%{x:.1f} Km<br>'+proxy_label+'=%{y:.1f}<br>'+formula+'<extra>'+name+'</extra>'
            ))
    marker_color = plot[color_by] if color_by in plot else None
    marker_numeric = pd.to_numeric(marker_color, errors='coerce') if marker_color is not None else None
    if marker_numeric is not None and marker_numeric.notna().any():
        marker = dict(size=point_size, line=dict(color='black', width=0.6))
        residual_like = color_by in ['Residual_km','Delta_km'] or str(color_by).lower().startswith(('residual','delta'))
        marker.update(
            color=marker_numeric,
            colorscale=RESIDUAL_COLORSCALE if residual_like else 'Viridis',
            colorbar=dict(title=local_option_label(color_by)),
        )
        if residual_like:
            marker['cmid'] = 0
        custom = np.c_[
            plot[x_col].round(1),
            pd.to_numeric(plot[proxy], errors='coerce').round(1),
            marker_numeric.round(1) if color_by in plot else np.repeat('', len(plot))
        ]
        fig.add_trace(go.Scatter(
            x=pd.to_numeric(plot[x_col], errors='coerce'),
            y=pd.to_numeric(plot[proxy], errors='coerce'),
            mode='markers', name='Blind validation samples',
            marker=marker, customdata=custom,
            hovertemplate=x_label+'=%{customdata[0]:.1f}<br>'+proxy_label+'=%{customdata[1]:.1f}<br>'+local_option_label(color_by)+'=%{customdata[2]:.1f}<extra></extra>'
        ))
    else:
        if color_by in plot:
            palette = px.colors.qualitative.Set2
            groups = [(str(k), g) for k, g in plot.groupby(plot[color_by].astype(str), dropna=False)]
        else:
            palette = ['#e45756']
            groups = [('Blind validation samples', plot)]
        for i, (name, g) in enumerate(groups):
            custom = np.c_[
                g[x_col].round(1),
                pd.to_numeric(g[proxy], errors='coerce').round(2),
                np.repeat(name, len(g))
            ]
            fig.add_trace(go.Scatter(
                x=pd.to_numeric(g[x_col], errors='coerce'),
                y=pd.to_numeric(g[proxy], errors='coerce'),
                mode='markers', name=name,
                marker=dict(size=point_size, color=palette[i % len(palette)], line=dict(color='black', width=0.6)),
                customdata=custom,
                hovertemplate=x_label+'=%{customdata[0]:.1f}<br>'+proxy_label+'=%{customdata[1]:.1f}<br>%{customdata[2]}<extra></extra>'
            ))
    fig = add_binned_summary_traces(fig, plot, x_col, proxy, trend_stat, trend_bin_width, trend_min_n, None, 'moving')
    fig.update_layout(
        template='plotly_white', height=430,
        margin=dict(l=20,r=20,t=20,b=20),
        xaxis_title=x_label,
        yaxis_title=proxy_label,
        legend=dict(orientation='h', y=-0.24, x=0)
    )
    fig.update_xaxes(range=x_range if x_range else None, showgrid=True)
    fig.update_yaxes(range=y_range if y_range else None, rangemode='tozero' if not y_range else None, showgrid=True)
    return fig, (formula_lookup if show_formulas else {})

PROXY_THICKNESS_LABELS = {
    'H_Profeta2015_SrY_km':'Profeta Sr/Y',
    'H_Sundell2021_SrY_km':'Sundell Sr/Y',
    'H_Zou2021_SrY_SVRE_km':'Zou Sr/Y',
    'H_Hu2017_Collisional_SrY_km':'Hu collisional Sr/Y',
    'H_Profeta2015_LaYbN_km':'Profeta La/Yb(N)',
    'H_Sundell2021_LaYbN_km':'Sundell La/Yb(N)',
    'H_Zou2021_LaYbN_SVRE_km':'Zou La/Yb(N)',
    'H_Hu2017_Collisional_LaYbN_km':'Hu collisional La/Yb(N)',
    'H_Mantle2008_CeY_sample_km':'Mantle & Collins Ce/Y',
    'H_Dhuime2015_RbSr_km':'Dhuime Rb/Sr',
    'H_Dhuime2015_SiO2_km':'Dhuime SiO2',
    'H_FarnerLee2017_SiO2_ElevMoho_km':'Farner-Lee SiO2 elevation-Moho',
    'H_FarnerLee2017_FeOt_ElevMoho_km':'Farner-Lee FeOt elevation-Moho',
    'H_FarnerLee2017_CaO_ElevMoho_km':'Farner-Lee CaO elevation-Moho',
    'H_FarnerLee2017_K2O_ElevMoho_km':'Farner-Lee K2O elevation-Moho',
    'H_LuffiDucea2022_Elevation_Moho_km':'Luffi-Ducea elevation-Moho',
    'H_Sundell2021_Paired_km':'Sundell paired Sr/Y + La/Yb(N)',
    'H_GAME_LuffiDucea2022_km':'GAME consensus',
}
PROXY_THICKNESS_FORMULAS = {
    'H_Profeta2015_SrY_km':'H = (Sr/Y + 7.25) / 0.90',
    'H_Sundell2021_SrY_km':'H = 19.6 ln(Sr/Y) - 24',
    'H_Zou2021_SrY_SVRE_km':'H = 1.11(Sr/Y) + 8.05',
    'H_Hu2017_Collisional_SrY_km':'H = 0.67(Sr/Y) + 28.21',
    'H_Profeta2015_LaYbN_km':'H = ln(La/Yb(N) / 0.98) / 0.047',
    'H_Sundell2021_LaYbN_km':'H = 17 ln(La/Yb(N)) + 6.9',
    'H_Zou2021_LaYbN_SVRE_km':'H = 21.277 ln(1.0204 La/Yb(N))',
    'H_Hu2017_Collisional_LaYbN_km':'H = 27.78 ln(0.34 La/Yb(N))',
    'H_Mantle2008_CeY_sample_km':'H = 18.0505 ln(Ce/Y) + 21.5587',
    'H_Dhuime2015_RbSr_km':'H = 426.8(Rb/Sr) + 4.1',
    'H_Dhuime2015_SiO2_km':'H = 3.5(SiO2 wt%) - 157.8',
    'H_FarnerLee2017_SiO2_ElevMoho_km':'h = 0.81(SiO2 wt%) - 45.82; H = 6.79h + 26.40',
    'H_FarnerLee2017_FeOt_ElevMoho_km':'h = -1.70(FeOt wt%) + 12.78; H = 6.79h + 26.40',
    'H_FarnerLee2017_CaO_ElevMoho_km':'h = -1.57(CaO wt%) + 12.18; H = 6.79h + 26.40',
    'H_FarnerLee2017_K2O_ElevMoho_km':'h = 3.27(K2O wt%) - 4.54; H = 6.79h + 26.40',
    'H_LuffiDucea2022_Elevation_Moho_km':'H = 6.79(elevation km) + 26.40',
    'H_Sundell2021_Paired_km':'H = 10.3 ln(Sr/Y) + 8.8 ln(La/Yb(N)) - 10.6',
    'H_GAME_LuffiDucea2022_km':'GAME consensus; usable mohometer count varies by sample after availability + MAD filtering',
}

PROXY_LIBRARY = [
    {'column':'H_Profeta2015_SrY_km','category':'Arc trace-element proxies','kind':'single','inputs':['Sr','Y','Sr_Y']},
    {'column':'H_Sundell2021_SrY_km','category':'Arc trace-element proxies','kind':'single','inputs':['Sr','Y','Sr_Y']},
    {'column':'H_Zou2021_SrY_SVRE_km','category':'Arc trace-element proxies','kind':'single','inputs':['Sr','Y','Sr_Y']},
    {'column':'H_Profeta2015_LaYbN_km','category':'Arc trace-element proxies','kind':'single','inputs':['La','Yb','La_Yb_N']},
    {'column':'H_Sundell2021_LaYbN_km','category':'Arc trace-element proxies','kind':'single','inputs':['La','Yb','La_Yb_N']},
    {'column':'H_Zou2021_LaYbN_SVRE_km','category':'Arc trace-element proxies','kind':'single','inputs':['La','Yb','La_Yb_N']},
    {'column':'H_Mantle2008_CeY_sample_km','category':'Arc trace-element proxies','kind':'single','inputs':['Ce','Y','Ce_Y']},
    {'column':'H_Hu2017_Collisional_SrY_km','category':'Collisional-belt proxies','kind':'single','inputs':['Sr','Y','Sr_Y']},
    {'column':'H_Hu2017_Collisional_LaYbN_km','category':'Collisional-belt proxies','kind':'single','inputs':['La','Yb','La_Yb_N']},
    {'column':'H_Sundell2021_Paired_km','category':'Paired / consensus methods','kind':'paired','inputs':['Sr_Y','La_Yb_N']},
    {'column':'H_GAME_LuffiDucea2022_km','category':'Paired / consensus methods','kind':'consensus','inputs':['MgO','GAME mohometers']},
    {'column':'H_Dhuime2015_RbSr_km','category':'Dhuime / silica proxies','kind':'single','inputs':['Rb','Sr','Rb_Sr']},
    {'column':'H_Dhuime2015_SiO2_km','category':'Dhuime / silica proxies','kind':'single','inputs':['SiO2']},
    {'column':'H_FarnerLee2017_SiO2_ElevMoho_km','category':'Elevation / topographic proxies','kind':'single','inputs':['SiO2']},
    {'column':'H_FarnerLee2017_FeOt_ElevMoho_km','category':'Elevation / topographic proxies','kind':'single','inputs':['FeO']},
    {'column':'H_FarnerLee2017_CaO_ElevMoho_km','category':'Elevation / topographic proxies','kind':'single','inputs':['CaO']},
    {'column':'H_FarnerLee2017_K2O_ElevMoho_km','category':'Elevation / topographic proxies','kind':'single','inputs':['K2O']},
    {'column':'H_LuffiDucea2022_Elevation_Moho_km','category':'Elevation / topographic proxies','kind':'single','inputs':['Elevation_km']},
]
PROXY_CATEGORY_ORDER = ['Paired / consensus methods','Arc trace-element proxies','Collisional-belt proxies','Dhuime / silica proxies','Elevation / topographic proxies']
PROXY_VALUE_LIBRARY = {
    'Sr_Y':{'label':'Sr/Y','category':'Arc and collisional ratios'},
    'La_Yb_N':{'label':'La/Yb(N)','category':'Arc and collisional ratios'},
    'Ce_Y':{'label':'Ce/Y','category':'Arc and mafic ratios'},
    'La_Y':{'label':'La/Y','category':'Routine-ratio diagnostics'},
    'Rb_Sr':{'label':'Rb/Sr','category':'Dhuime / silica proxies'},
    'SiO2':{'label':'SiO2','category':'Dhuime / silica proxies'},
    'Elevation_km':{'label':'Elevation [km]','category':'Elevation / topographic proxies'},
    'Dy_Yb':{'label':'Dy/Yb','category':'Mafic / HREE diagnostics'},
    'Gd_Yb':{'label':'Gd/Yb','category':'Mafic / HREE diagnostics'},
    'Zr_Ti':{'label':'Zr/Ti','category':'Mafic / immobile diagnostics'},
}

def proxy_library_category_map():
    return {item['column']:item['category'] for item in PROXY_LIBRARY}

def proxy_library_methods_available(df):
    return [item['column'] for item in PROXY_LIBRARY if has_numeric_column(df, item['column'])]

def proxy_library_multiselect(options, default=None, key='proxy_library_methods'):
    default = default or []
    # Order options by PROXY_LIBRARY canonical order, then any extras
    ordered = [item['column'] for item in PROXY_LIBRARY if item['column'] in options]
    extras  = [c for c in options if c not in ordered]
    all_opts = ordered + extras
    chosen = st.multiselect(
        'Proxy thickness methods',
        all_opts,
        default=[c for c in default if c in all_opts] or all_opts[:3],
        key=key,
        format_func=lambda c: PROXY_THICKNESS_LABELS.get(c, c),
        label_visibility='collapsed',
    )
    return list(dict.fromkeys(chosen))

def proxy_value_options_available(df):
    return [c for c in PROXY_VALUE_LIBRARY if has_numeric_column(df, c)]

def proxy_value_label(c):
    return PROXY_VALUE_LIBRARY.get(c,{}).get('label',str(c).replace('_','/'))

def proxy_value_curve_options(df):
    return [c for c in proxy_value_options_available(df) if curves(c)[1]]

def proxy_value_category_options(options):
    categories = []
    for c in options:
        cat = PROXY_VALUE_LIBRARY.get(c,{}).get('category','Other proxy values')
        if cat not in categories:
            categories.append(cat)
    return categories

# ─── Unified XY graph (Validate + Predict) ───────────────────────────────────
# Three dropdowns — X, Y, Colour — over a flat list of every numeric/categorical
# column on the bench. Default layout matches the classic predicted-vs-known
# scatter: X = Observed_km, Y = Predicted_km, Colour = Residual_km on the
# red-cream-blue residual ramp. Pick any other proxy/element column for any
# axis and the scatter follows. Replaces the half-dozen near-duplicate
# proxy/GAME chart blocks while keeping the proxy formula overlay when
# the X column matches a known calibration input.

_RESIDUAL_COLOR_COLS = {'Residual_km', 'Delta_km', 'GAME_delta_km'}


# ── Taxonomy for the Validation-plot axis dropdowns ──────────────────────
# 12 categories covering everything that lives on a typical bench. Order
# in this list IS the dropdown sort order — estimates first (most likely
# to be picked), then ratios + raw chemistry, then metadata.
_SXY_CATEGORY_ORDER = [
    'Estimate: ML model',         # Predicted_km from a trained model
    'Estimate: Multi-ratio proxy',# Sundell paired, GAME consensus
    'Estimate: Single-ratio proxy',# Profeta Sr/Y, Sundell Sr/Y, Mantle Ce/Y, …
    'Estimate: Reference',        # Observed_km, CRUST1, LithoRef18
    'Residual',                   # Residual_km, Delta_km, GAME_delta_km
    'Value: Ratio',               # Sr_Y, La_Yb_N, Ce_Y, Rb_Sr, …
    'Value: Major oxide',         # SiO2, TiO2, Al2O3, FeO, MgO, CaO, …
    'Value: Trace element',       # ppm-scale: Sr, Y, La, Yb, REE, PGE, …
    'Value: Isotope',             # Sr87/Sr86, Nd143/Nd144, εNd, He3/He4 …
    'Value: Volatile',            # H2O, LOI, CO2, F, Cl, S, …
    'Location / age',             # Lat, Lon, Age_Ma, Elevation_km
    'Category',                   # Tectonic_Setting, Group_Name, Model, …
]
_SXY_CATEGORY_INDEX = {c: i for i, c in enumerate(_SXY_CATEGORY_ORDER)}

# Multi-ratio thickness columns (paired / consensus methods)
_SXY_MULTI_RATIO_COLS = {'H_Sundell2021_Paired_km', 'H_GAME_LuffiDucea2022_km'}
_SXY_REFERENCE_COLS   = {'Observed_km', 'CRUST1_Total_Crust_km',
                          'CRUST1_Crystalline_Crust_km', 'CRUST1_Sediment_km',
                          'CRUST1_Benchmark_km',
                          'LithoRef18_Total_Crust_km', 'LithoRef18_LAB_km'}
_SXY_MAJOR_OXIDES = {'SiO2','TiO2','Al2O3','FeO','FeOt','Fe2O3','MnO','MgO','CaO',
                     'Na2O','K2O','P2O5','B2O3','Cr2O3','NiO'}
_SXY_TRACE_ELEMENTS = {
    # REE + adjacent
    'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Y',
    # LILE
    'Rb','Sr','Cs','Ba','Pb',
    # HFSE + actinides
    'Zr','Hf','Nb','Ta','Th','U',
    # First-row + adjacent transition
    'Sc','V','Cr','Mn','Co','Ni','Cu','Zn','Ga',
    # Light + post-transition + metalloids
    'Li','Be','B','Ge','As','Se','Br','I',
    # Refractory + PGE
    'Mo','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','W','Re','Os','Ir','Pt',
    'Au','Hg','Tl','Bi',
}
_SXY_VOLATILES = {'H2O','H2O_total','H2O_plus','H2O_minus','LOI','CO2','CO',
                   'F','Cl','OH','CH4','SO2','SO3','SO4','S','Volatiles_total',
                   'O','Others','Cl2'}


def simple_xy_column_category(col):
    """Bucket a bench column name into one of 12 axis-picker categories.

    The taxonomy is biased toward the actual decisions a user is making
    when building a validation plot: pick an estimate (ML / multi-ratio
    proxy / single-ratio proxy / reference) on one axis and a value
    (ratio, oxide, trace, isotope, volatile, location, age) on the other.
    Categories are emitted as a string so they can be used directly in
    the picker's ``format_func`` as a prefix label.

    ``Predicted_km`` is the *active-model* prediction in the bench. Its
    category is honest about the source: in ratio / GAME mode it carries
    the proxy estimate (not an ML output) so we route it accordingly.

    ``Predicted_<safe_name>_km`` columns are per-model wide projections
    of the long-format bench — each one is a specific trained ML model's
    prediction surfaced as its own axis option.

    Falls back to ``'Category'`` for anything we can't classify (Group_Name,
    Model, Tectonic_Setting, Arc_or_Segment, …) — i.e. metadata columns
    are sensibly grouped at the end of the dropdown.
    """
    s = str(col)

    # Estimates ──────────────────────────────────────────────────────────
    if s == 'Predicted_km':
        # Honour what actually populated Predicted_km — ML mode uses the
        # model's .predict(), ratio modes synthesise from a proxy column.
        try:
            _pmt = st.session_state.get('primary_model_type', 'Machine Learning')
        except Exception:
            _pmt = 'Machine Learning'
        if _pmt == 'Multi-ratio':
            return 'Estimate: Multi-ratio proxy'
        if _pmt == 'Single-ratio':
            return 'Estimate: Single-ratio proxy'
        return 'Estimate: ML model'
    # Per-model wide columns produced by benchmark_uploaded() — pattern is
    # Predicted_<sanitised_name>_km. Always ML model output.
    if s.startswith('Predicted_') and s.endswith('_km') and s != 'Predicted_km':
        return 'Estimate: ML model'
    if s in _SXY_REFERENCE_COLS:
        return 'Estimate: Reference'
    if s in _RESIDUAL_COLOR_COLS:           # Residual_km, Delta_km, GAME_delta_km
        return 'Residual'
    # Per-model residual columns
    if s.startswith('Residual_') and s.endswith('_km') and s != 'Residual_km':
        return 'Residual'
    if s in _SXY_MULTI_RATIO_COLS:
        return 'Estimate: Multi-ratio proxy'
    # Anything else starting with H_ and ending with _km is a single-
    # ratio proxy thickness (Profeta, Sundell single, Zou single, Hu,
    # Dhuime, Farner-Lee, Luffi elevation, etc.)
    if s.startswith('H_') and s.endswith('_km'):
        return 'Estimate: Single-ratio proxy'

    # Values ─────────────────────────────────────────────────────────────
    # Order matters: handle the more specific buckets (Major oxide,
    # Trace element, Volatile, Location) BEFORE PROXY_VALUE_LIBRARY,
    # because that constant includes a few non-ratio entries (SiO2,
    # Elevation_km) that should land in their own buckets.
    if s in _SXY_MAJOR_OXIDES:
        return 'Value: Major oxide'
    if s in _SXY_TRACE_ELEMENTS or s.endswith('_ppm'):
        return 'Value: Trace element'
    if s in _SXY_VOLATILES:
        return 'Value: Volatile'
    if s in {'Lat','Lon','Latitude','Longitude','Age_Ma','Elevation_km',
             'GMRT_elevation_km'}:
        return 'Location / age'
    if s in PROXY_VALUE_LIBRARY:            # Sr_Y, La_Yb_N, Ce_Y, Rb_Sr, …
        return 'Value: Ratio'
    if s in {'La_Yb_raw','La_Y','MnO_MgO','Dy_Yb','Gd_Yb','Eu_Eu_star','Rb_Sr'}:
        return 'Value: Ratio'
    # Isotope detection — covers spelled-out forms used by enrich() +
    # raw Luffi T1 forms before the rename pass would have converted them.
    if (s.startswith('Nd143') or s.startswith('Sr87') or s.startswith('Pb20')
            or s.startswith('Hf176') or s.startswith('Os1') or s.startswith('Os')
            or s.startswith('Re187') or s.startswith('He3') or s.startswith('He4')
            or s.startswith('K40') or s.startswith('Ar40')
            or s == 'epsilon_Nd' or s == 'He_R_over_Ra'):
        return 'Value: Isotope'

    # Fallback: anything not classified is metadata / categorical
    return 'Category'


def simple_xy_axis_options(bench_df):
    """Flat list of selectable columns for the X / Y / Colour dropdowns.

    Numerics first (the useful ones for scatter plots), then categoricals
    appended at the end. Any column with at least one non-null value is
    eligible; all-NaN columns are dropped so the picker doesn't dangle
    options that won't render anything.

    A small canonical order is honoured at the front (Observed_km,
    Predicted_km, Residual_km, common ratios + petrologic predictors)
    so the most-used choices are at the top of the dropdown rather than
    sorted alphabetically into the middle.
    """
    if bench_df is None or bench_df.empty:
        return []
    preferred = [
        'Observed_km', 'Predicted_km', 'Residual_km',
        'Sr_Y', 'La_Yb_N', 'Ce_Y', 'Dy_Yb', 'Gd_Yb', 'Rb_Sr',
        'SiO2', 'MgO', 'CaO', 'K2O', 'TiO2', 'Al2O3', 'FeO', 'Na2O',
        'Age_Ma', 'Elevation_km',
        'H_GAME_LuffiDucea2022_km', 'H_Sundell2021_Paired_km',
        'H_Profeta2015_SrY_km', 'H_Sundell2021_SrY_km', 'H_Zou2021_SrY_SVRE_km',
        'H_Profeta2015_LaYbN_km', 'H_Sundell2021_LaYbN_km', 'H_Zou2021_LaYbN_SVRE_km',
        'CRUST1_Total_Crust_km', 'LithoRef18_Total_Crust_km',
        'Tectonic_Setting', 'Arc_or_Segment', 'Geologic_Domain',
        'Geologic_Era', 'Geologic_Period', 'Rock_Type_Model',
        'Model', 'Algorithm', 'Group_Name',
    ]
    out = []
    seen = set()
    for c in preferred:
        if c in bench_df.columns and c not in seen:
            if bench_df[c].notna().any():
                out.append(c); seen.add(c)
    # Append any other useful columns not already covered
    for c in bench_df.columns:
        if c in seen or str(c).startswith('_'):
            continue
        if bench_df[c].notna().any():
            out.append(c); seen.add(c)
    return out


def _simple_xy_formula_curve(x_col, y_col, x_grid):
    """Analytic proxy formula y(x), but only for the (X, Y) pairs where the
    proxy's input axis matches X. Returns None for unrelated pairs so we
    don't draw nonsense curves. Mirrors the formulas in ``enrich``.
    """
    if x_grid is None or len(x_grid) == 0:
        return None
    x = np.asarray(x_grid, dtype=float)
    safe_log = lambda v: np.log(np.where(v > 0, v, np.nan))
    table = {
        ('Sr_Y', 'H_Profeta2015_SrY_km'):           lambda x: (x + 7.25) / 0.90,
        ('Sr_Y', 'H_Sundell2021_SrY_km'):           lambda x: 19.6 * safe_log(x) - 24,
        ('Sr_Y', 'H_Zou2021_SrY_SVRE_km'):          lambda x: 1.11 * x + 8.05,
        ('Sr_Y', 'H_Hu2017_Collisional_SrY_km'):    lambda x: 0.67 * x + 28.21,
        ('La_Yb_N', 'H_Profeta2015_LaYbN_km'):      lambda x: safe_log(x / 0.98) / 0.047,
        ('La_Yb_N', 'H_Sundell2021_LaYbN_km'):      lambda x: 17.0 * safe_log(x) + 6.9,
        ('La_Yb_N', 'H_Zou2021_LaYbN_SVRE_km'):     lambda x: 21.277 * safe_log(1.0204 * x),
        ('La_Yb_N', 'H_Hu2017_Collisional_LaYbN_km'): lambda x: 27.78 * safe_log(0.34 * x),
        ('Ce_Y',  'H_Mantle2008_CeY_sample_km'):    lambda x: 18.0505 * safe_log(x) + 21.5587,
        ('Rb_Sr', 'H_Dhuime2015_RbSr_km'):          lambda x: 426.8 * x + 4.1,
        ('SiO2',  'H_Dhuime2015_SiO2_km'):          lambda x: 3.5 * x - 157.8,
        ('SiO2',  'H_FarnerLee2017_SiO2_ElevMoho_km'): lambda x: 6.79 * (0.81 * x - 45.82) + 26.40,
        ('FeO',   'H_FarnerLee2017_FeOt_ElevMoho_km'): lambda x: 6.79 * (-1.70 * x + 12.78) + 26.40,
        ('CaO',   'H_FarnerLee2017_CaO_ElevMoho_km'):  lambda x: 6.79 * (-1.57 * x + 12.18) + 26.40,
        ('K2O',   'H_FarnerLee2017_K2O_ElevMoho_km'):  lambda x: 6.79 * (3.27 * x - 4.54) + 26.40,
        ('Elevation_km', 'H_LuffiDucea2022_Elevation_Moho_km'): lambda x: 6.79 * x + 26.40,
    }
    fn = table.get((x_col, y_col))
    return fn(x) if fn else None


def simple_xy_figure(bench_df, x_col, y_col, color_col=None,
                     show_one_to_one=True, show_curve=True, show_best_fit=False,
                     point_size=6, height=520,
                     trend_stat='Off', trend_bin_width=5.0, trend_min_n=5,
                     clip_pct=0.0,
                     show_points=True, group_stat='Off', group_col='Group_Name'):
    """Plain X-vs-Y scatter coloured by any third column.

    Colour mode auto-detects:
      • Residual-like column (Residual_km / Delta_km / GAME_delta_km) →
        red-cream-blue ``RESIDUAL_COLORSCALE`` centred on zero.
      • Other numeric column → continuous Viridis ramp.
      • Categorical / object column → per-category discrete colours.
      • ``None`` → single-tone grey.

    Optional overlays:
      • **1:1 line** when both axes are crustal-thickness columns.
      • **Formula curve** when (X, Y) matches a known proxy calibration
        input pair (e.g. X=Sr_Y, Y=H_Profeta2015_SrY_km).
      • **Best fit** least-squares line.
      • **Moving summary** — binned median / mean / both, drawn as a
        line over the X domain with the bin width and minimum-N
        chosen by the user. Lifted from the legacy proxy plot so the
        Validation plot can replace it 1:1.
      • **Outlier clip** — trim X and Y by the chosen percentile on
        both tails before plotting. Stops a handful of off-scale
        outliers from compressing the rest of the chart.

    ``trend_stat``: ``'Off'`` (default), ``'Median'``, ``'Mean'``, or ``'Both'``.
    ``clip_pct``: 0 (no clip), 0.5, 1, 2, 5 — % trimmed from each tail.
    """
    if bench_df is None or bench_df.empty:
        return None
    if x_col not in bench_df.columns or y_col not in bench_df.columns:
        return None

    xs = pd.to_numeric(bench_df[x_col], errors='coerce')
    ys = pd.to_numeric(bench_df[y_col], errors='coerce')
    ok = xs.notna() & ys.notna()
    if not ok.any():
        return None

    # Outlier clipping — symmetric percentile trim on both axes. Only
    # filters point inclusion; doesn't affect the X domain used for
    # formula curves or the 1:1 line, those still span the data.
    if clip_pct and clip_pct > 0:
        _xs_ok = xs[ok]; _ys_ok = ys[ok]
        _q = float(clip_pct) / 100.0
        _xlo, _xhi = float(_xs_ok.quantile(_q)), float(_xs_ok.quantile(1 - _q))
        _ylo, _yhi = float(_ys_ok.quantile(_q)), float(_ys_ok.quantile(1 - _q))
        ok = ok & xs.between(_xlo, _xhi) & ys.between(_ylo, _yhi)
        if not ok.any():
            return None

    # Decide colour treatment
    is_residual = color_col in _RESIDUAL_COLOR_COLS
    is_numeric_color = (color_col is not None
                        and color_col in bench_df.columns
                        and pd.api.types.is_numeric_dtype(bench_df[color_col]))
    is_categorical_color = (color_col is not None
                            and color_col in bench_df.columns
                            and not is_numeric_color)

    custom_cols = ['Model'] if 'Model' in bench_df.columns else []
    if 'Group_Name' in bench_df.columns and 'Group_Name' not in custom_cols:
        custom_cols.append('Group_Name')

    plot = bench_df.loc[ok, list(dict.fromkeys([x_col, y_col]
                                                + ([color_col] if color_col else [])
                                                + custom_cols))].copy()
    plot[x_col] = pd.to_numeric(plot[x_col], errors='coerce')
    plot[y_col] = pd.to_numeric(plot[y_col], errors='coerce')

    px_kwargs = dict(x=x_col, y=y_col, template='plotly_white',
                     custom_data=custom_cols if custom_cols else None)
    if color_col and color_col in plot.columns:
        px_kwargs['color'] = color_col
        if is_residual:
            px_kwargs['color_continuous_scale']    = RESIDUAL_COLORSCALE
            px_kwargs['color_continuous_midpoint'] = 0
        elif is_numeric_color:
            px_kwargs['color_continuous_scale'] = THICKNESS_COLORSCALE if str(color_col).endswith('_km') else 'Viridis'
        # Categorical → px.scatter handles it automatically with discrete palette

    fig = px.scatter(plot, **px_kwargs)
    # Sample-point opacity / visibility — when show_points is False we
    # keep the trace in the legend (so the user can flip it back on with
    # one click) but mark it ``legendonly`` so the scatter is hidden.
    # That's the right behaviour when the user wants only the
    # group-medians or moving summary to be visible.
    _scatter_visible = True if show_points else 'legendonly'
    fig.update_traces(marker=dict(size=int(point_size),
                                  line=dict(color='rgba(17,24,39,0.45)', width=0.4)),
                      visible=_scatter_visible)

    # Hover: include Model + Group_Name when available
    _hover_lines = [f'<b>{x_col}=%{{x:.3g}}</b>',
                    f'{y_col}=%{{y:.3g}}']
    if 'Model' in custom_cols:
        _hover_lines.append('Model=%{customdata[0]}')
    if 'Group_Name' in custom_cols:
        _hover_lines.append(f'Group=%{{customdata[{custom_cols.index("Group_Name")}]}}')
    fig.update_traces(hovertemplate='<br>'.join(_hover_lines) + '<extra></extra>')

    # 1:1 line — only meaningful when both axes are thickness columns
    _km_cols = {'Observed_km', 'Predicted_km'} | set(PROXY_THICKNESS_LABELS.keys())
    if show_one_to_one and x_col in _km_cols and y_col in _km_cols:
        _x_max = float(np.nanmax(plot[x_col].to_numpy()))
        _y_max = float(np.nanmax(plot[y_col].to_numpy()))
        _ax_max = max(_x_max, _y_max, 1.0)
        fig.add_trace(go.Scatter(
            x=[0, _ax_max], y=[0, _ax_max], mode='lines',
            name='1:1', line=dict(color='black', dash='dash', width=1.2),
            hoverinfo='skip', showlegend=True,
        ))

    # Proxy formula curve — only when (X, Y) matches a known calibration pair
    if show_curve:
        _x_lo = float(np.nanmin(plot[x_col].to_numpy()))
        _x_hi = float(np.nanmax(plot[x_col].to_numpy()))
        _x_pad = max(0.05 * (_x_hi - _x_lo), 1e-3)
        _x_grid = np.linspace(max(0.0, _x_lo - _x_pad), _x_hi + _x_pad, 200)
        _curve_y = _simple_xy_formula_curve(x_col, y_col, _x_grid)
        if _curve_y is not None and np.isfinite(_curve_y).any():
            _ok_curve = np.isfinite(_curve_y)
            fig.add_trace(go.Scatter(
                x=_x_grid[_ok_curve], y=_curve_y[_ok_curve], mode='lines',
                name=f'{PROXY_THICKNESS_LABELS.get(y_col, y_col)} formula',
                line=dict(color='black', width=2.0, dash='dot'),
                hovertemplate=f'<b>formula</b><br>{x_col}=%{{x:.2f}} → {y_col}=%{{y:.1f}}<extra></extra>',
            ))
            _formula_text = PROXY_THICKNESS_FORMULAS.get(y_col, '')
            if _formula_text:
                fig.add_annotation(x=0.99, y=0.02, xref='paper', yref='paper',
                                   text=f'<i>{_formula_text}</i>',
                                   showarrow=False, xanchor='right', yanchor='bottom',
                                   font=dict(size=10, color='#444'),
                                   bgcolor='rgba(255,255,255,0.78)')

    # Best-fit (single line over all points)
    if show_best_fit and ok.sum() >= 2:
        try:
            slope, intercept = np.polyfit(plot[x_col].to_numpy(), plot[y_col].to_numpy(), 1)
            _xfit = np.linspace(float(plot[x_col].min()), float(plot[x_col].max()), 50)
            _yfit = slope * _xfit + intercept
            fig.add_trace(go.Scatter(
                x=_xfit, y=_yfit, mode='lines',
                name=f'Best fit (slope={slope:.2f})',
                line=dict(color='#1e3a5f', width=1.6),
                hovertemplate=(f'<b>Best fit</b><br>slope={slope:.3g}, '
                               f'intercept={intercept:.3g}<extra></extra>'),
            ))
        except Exception:
            pass

    # Moving summary — bin X by the chosen width, take median or mean of
    # Y per bin (skipping bins with fewer than ``trend_min_n`` points), and
    # draw the resulting curve. Lifted from the legacy proxy plots so the
    # Validation plot is a 1:1 replacement for them.
    if str(trend_stat) in ('Median', 'Mean', 'Both') and ok.sum() >= 2:
        try:
            _bw = max(0.1, float(trend_bin_width))
            _x_arr = plot[x_col].to_numpy(dtype=float)
            _y_arr = plot[y_col].to_numpy(dtype=float)
            _x_min = float(np.nanmin(_x_arr))
            _x_max = float(np.nanmax(_x_arr))
            if _x_max > _x_min:
                _edges = np.arange(_x_min, _x_max + _bw, _bw)
                if len(_edges) >= 2:
                    _idx = np.digitize(_x_arr, _edges) - 1
                    _idx = np.clip(_idx, 0, len(_edges) - 2)
                    _bin_centres = (_edges[:-1] + _edges[1:]) / 2.0
                    _med_y, _mean_y, _ns = [], [], []
                    for _b in range(len(_bin_centres)):
                        _mask = _idx == _b
                        _n_in_bin = int(np.sum(_mask & ~np.isnan(_y_arr)))
                        if _n_in_bin >= int(trend_min_n):
                            _med_y.append(float(np.nanmedian(_y_arr[_mask])))
                            _mean_y.append(float(np.nanmean(_y_arr[_mask])))
                        else:
                            _med_y.append(np.nan); _mean_y.append(np.nan)
                        _ns.append(_n_in_bin)
                    if str(trend_stat) in ('Median', 'Both'):
                        fig.add_trace(go.Scatter(
                            x=_bin_centres, y=_med_y, mode='lines+markers',
                            name=f'Moving median (bin {_bw:g}, min N {int(trend_min_n)})',
                            line=dict(color='#1e3a5f', width=2.5),
                            marker=dict(size=6, color='#1e3a5f',
                                        line=dict(color='white', width=1.0)),
                            hovertemplate=(f'<b>median</b><br>{x_col} bin centre=%{{x:.2f}}<br>'
                                           f'{y_col}=%{{y:.2f}}<extra></extra>'),
                        ))
                    if str(trend_stat) in ('Mean', 'Both'):
                        fig.add_trace(go.Scatter(
                            x=_bin_centres, y=_mean_y, mode='lines+markers',
                            name=f'Moving mean (bin {_bw:g}, min N {int(trend_min_n)})',
                            line=dict(color='#b91c1c', width=2.0, dash='dot'),
                            marker=dict(size=6, color='#b91c1c',
                                        line=dict(color='white', width=1.0)),
                            hovertemplate=(f'<b>mean</b><br>{x_col} bin centre=%{{x:.2f}}<br>'
                                           f'{y_col}=%{{y:.2f}}<extra></extra>'),
                        ))
        except Exception:
            pass

    # Per-group medians / means — drawn LAST so they paint on top of
    # everything else and stay legible even when the scatter is dense.
    # One marker per group at (group median X, group median Y) for
    # Median; (mean X, mean Y) for Mean. Both can be enabled via
    # ``group_stat='Both'``. Each marker carries the group name in
    # customdata so the hover stays informative.
    if (str(group_stat) in ('Median', 'Mean', 'Both')
            and group_col in bench_df.columns
            and bench_df[group_col].notna().any()):
        try:
            _g_src = bench_df.loc[ok, [x_col, y_col, group_col]].copy()
            _g_src[x_col] = pd.to_numeric(_g_src[x_col], errors='coerce')
            _g_src[y_col] = pd.to_numeric(_g_src[y_col], errors='coerce')
            _g_src = _g_src.dropna(subset=[x_col, y_col, group_col])
            _palette_g = list(px.colors.qualitative.Bold) + list(px.colors.qualitative.Set2)
            _group_names = list(dict.fromkeys(_g_src[group_col].astype(str).tolist()))
            _grp_color_map = {
                _gn: _palette_g[_i % len(_palette_g)]
                for _i, _gn in enumerate(_group_names)
            }
            if str(group_stat) in ('Median', 'Both'):
                _med = (_g_src.groupby(group_col, dropna=True)
                              .agg(_x=(x_col, 'median'),
                                   _y=(y_col, 'median'),
                                   _n=(x_col, 'size')).reset_index())
                if not _med.empty:
                    _med_colors = [_grp_color_map.get(str(g), '#111827')
                                   for g in _med[group_col]]
                    fig.add_trace(go.Scatter(
                        x=_med['_x'], y=_med['_y'], mode='markers',
                        marker=dict(symbol='diamond', size=14,
                                    color=_med_colors,
                                    line=dict(color='black', width=1.4)),
                        name='Group median',
                        customdata=np.c_[_med[group_col].astype(str), _med['_n']],
                        hovertemplate=('<b>%{customdata[0]}</b><br>'
                                       'Group median<br>'
                                       f'{x_col}=%{{x:.2f}}<br>'
                                       f'{y_col}=%{{y:.2f}}<br>'
                                       'N=%{customdata[1]}<extra></extra>'),
                    ))
            if str(group_stat) in ('Mean', 'Both'):
                _avg = (_g_src.groupby(group_col, dropna=True)
                              .agg(_x=(x_col, 'mean'),
                                   _y=(y_col, 'mean'),
                                   _n=(x_col, 'size')).reset_index())
                if not _avg.empty:
                    _avg_colors = [_grp_color_map.get(str(g), '#111827')
                                   for g in _avg[group_col]]
                    fig.add_trace(go.Scatter(
                        x=_avg['_x'], y=_avg['_y'], mode='markers',
                        marker=dict(symbol='star', size=14,
                                    color=_avg_colors,
                                    line=dict(color='black', width=1.4)),
                        name='Group mean',
                        customdata=np.c_[_avg[group_col].astype(str), _avg['_n']],
                        hovertemplate=('<b>%{customdata[0]}</b><br>'
                                       'Group mean<br>'
                                       f'{x_col}=%{{x:.2f}}<br>'
                                       f'{y_col}=%{{y:.2f}}<br>'
                                       'N=%{customdata[1]}<extra></extra>'),
                    ))
        except Exception:
            pass

    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=x_col, yaxis_title=y_col,
        legend=dict(orientation='h', y=-0.18, x=0,
                    font=dict(size=9), bgcolor='rgba(255,255,255,0.6)'),
    )
    return fig


def _LEGACY_unified_orphan_BEGIN():
    """Build the menu of selectable methods grouped by category.

    Returns ``[(category, [(method_id, method_label, column, formula_text)])]``
    keyed by category. Methods are only listed when their value column has
    at least one non-null entry on this bench, so the picker doesn't dangle
    options that won't render.

    ``method_id`` is a stable string used as a series key. For ML methods
    it's ``'ml::<Model name>'`` and the column is ``Predicted_km`` filtered
    by ``Model == name``. For proxy/GAME methods it's ``'proxy::<col>'``
    and the column is read directly.
    """
    categories: dict = {}
    if bench_df is None or bench_df.empty:
        return categories

    # ── ML models ────────────────────────────────────────────────────────
    if has_observed and 'Model' in bench_df.columns and 'Predicted_km' in bench_df.columns:
        ml_models = [str(m) for m in dict.fromkeys(bench_df['Model'].dropna().astype(str).tolist())]
        if ml_models:
            categories['ML models (trained)'] = [
                (f'ml::{m}', m, 'Predicted_km',
                 'tree-ensemble fit; see Model tab for the feature list')
                for m in ml_models
            ]
    elif 'Predicted_km' in bench_df.columns:
        # Predict-tab path — one Predicted_km per Model (no Observed_km).
        ml_models = [str(m) for m in dict.fromkeys(bench_df['Model'].dropna().astype(str).tolist())] \
                    if 'Model' in bench_df.columns else ['(model)']
        if ml_models:
            categories['ML models (trained)'] = [
                (f'ml::{m}', m, 'Predicted_km',
                 'tree-ensemble fit; see Model tab for the feature list')
                for m in ml_models
            ]

    # ── Proxy / GAME methods (PROXY_LIBRARY drives the order) ────────────
    for cat in PROXY_CATEGORY_ORDER:
        items = []
        for entry in PROXY_LIBRARY:
            if entry['category'] != cat:
                continue
            col = entry['column']
            if col not in bench_df.columns:
                continue
            if pd.to_numeric(bench_df[col], errors='coerce').notna().sum() == 0:
                continue
            label   = PROXY_THICKNESS_LABELS.get(col, col)
            formula = PROXY_THICKNESS_FORMULAS.get(col, '')
            items.append((f'proxy::{col}', label, col, formula))
        if items:
            categories[cat] = items
    return categories


def unified_x_axis_options(bench_df, has_observed=True):
    """Columns the user can pick as the X axis of the unified graph.

    Preference order (when present): Observed_km (validation only), then
    common proxy ratios + petrologic predictors that make sense as an
    independent axis. Anything else with at least one numeric value is
    appended at the end so users can pick an unusual column if they have
    a reason.
    """
    if bench_df is None or bench_df.empty:
        return ['Observed_km'] if has_observed else ['Predicted_km']
    preferred = []
    if has_observed and 'Observed_km' in bench_df.columns:
        preferred.append('Observed_km')
    # Petrologic + ratio columns that are common X-axes for this kind of plot
    for c in ['Predicted_km', 'Sr_Y', 'La_Yb_N', 'Ce_Y', 'Dy_Yb', 'Gd_Yb',
              'Rb_Sr', 'SiO2', 'MgO', 'CaO', 'K2O', 'Age_Ma', 'Elevation_km',
              'Lat', 'Lon']:
        if c in bench_df.columns and c not in preferred:
            if pd.to_numeric(bench_df[c], errors='coerce').notna().any():
                preferred.append(c)
    return preferred


def _unified_formula_curve(col, x_col, x_grid, la_yb_mode='raw_ppm'):
    """Compute a method's predicted thickness across an X grid for the
    overlay curve. Returns y values aligned to ``x_grid``, or ``None`` when
    the formula doesn't depend on the chosen X axis (in which case we skip
    plotting a curve for that method-on-that-X combination).

    Only a handful of proxy↔X pairs make sense; anything else returns None
    to avoid plotting noise. The mapping is deliberately small — adding
    more (e.g. La/Y, Gd/Yb based proxies) is one new ``elif`` per pair.
    """
    if x_grid is None or len(x_grid) == 0:
        return None
    x = np.asarray(x_grid, dtype=float)
    safe_log = lambda v: np.log(np.where(v > 0, v, np.nan))
    # Sr/Y based
    if x_col == 'Sr_Y':
        if col == 'H_Profeta2015_SrY_km':            return (x + 7.25) / 0.90
        if col == 'H_Sundell2021_SrY_km':            return 19.6 * safe_log(x) - 24
        if col == 'H_Zou2021_SrY_SVRE_km':           return 1.11 * x + 8.05
        if col == 'H_Hu2017_Collisional_SrY_km':     return 0.67 * x + 28.21
    # La/Yb (chondrite-normalised) based
    if x_col == 'La_Yb_N':
        if col == 'H_Profeta2015_LaYbN_km':          return safe_log(x / 0.98) / 0.047
        if col == 'H_Sundell2021_LaYbN_km':          return 17.0 * safe_log(x) + 6.9
        if col == 'H_Zou2021_LaYbN_SVRE_km':         return 21.277 * safe_log(1.0204 * x)
        if col == 'H_Hu2017_Collisional_LaYbN_km':   return 27.78 * safe_log(0.34 * x)
    # Ce/Y based
    if x_col == 'Ce_Y' and col == 'H_Mantle2008_CeY_sample_km':
        return 18.0505 * safe_log(x) + 21.5587
    # Rb/Sr based
    if x_col == 'Rb_Sr' and col == 'H_Dhuime2015_RbSr_km':
        return 426.8 * x + 4.1
    # SiO2 based
    if x_col == 'SiO2':
        if col == 'H_Dhuime2015_SiO2_km':            return 3.5 * x - 157.8
        if col == 'H_FarnerLee2017_SiO2_ElevMoho_km':
            elev = 0.81 * x - 45.82
            return 6.79 * elev + 26.40
    # FeO based
    if x_col == 'FeO' and col == 'H_FarnerLee2017_FeOt_ElevMoho_km':
        elev = -1.70 * x + 12.78
        return 6.79 * elev + 26.40
    # CaO based
    if x_col == 'CaO' and col == 'H_FarnerLee2017_CaO_ElevMoho_km':
        elev = -1.57 * x + 12.18
        return 6.79 * elev + 26.40
    # K2O based
    if x_col == 'K2O' and col == 'H_FarnerLee2017_K2O_ElevMoho_km':
        elev = 3.27 * x - 4.54
        return 6.79 * elev + 26.40
    # Elevation based
    if x_col == 'Elevation_km' and col == 'H_LuffiDucea2022_Elevation_Moho_km':
        return 6.79 * x + 26.40
    return None


def unified_method_figure(bench_df, x_col, selected_method_ids,
                          show_one_to_one=True, show_curves=True,
                          show_medians=False, show_best_fit=False,
                          point_size=6, height=520):
    """Render the unified scatter — every selected method as its own series,
    plotted on the same axes against ``x_col``.

    Each series gets a stable colour from the qualitative palette (Bold +
    Set2). Optional overlays:

      • **1:1 line** — drawn when X = Observed_km, so ML/proxy/GAME points
        sit relative to the perfect-fit reference.
      • **Formula curves** — analytic proxy formulas plotted across the X
        domain, showing the calibration curve on top of its own scatter.
      • **Per-series median** — horizontal line at each method's median Y
        (lets the user eyeball which method consistently over- or
        under-predicts).
      • **Best-fit** — least-squares line per series, capped at the X range
        of that method's points.
    """
    if bench_df is None or bench_df.empty or not selected_method_ids:
        return None
    if x_col not in bench_df.columns:
        return None

    palette = list(px.colors.qualitative.Bold) + list(px.colors.qualitative.Set2)
    fig = go.Figure()

    # X grid for formula curves (covers the actual data range, padded 10%)
    x_vals_all = pd.to_numeric(bench_df[x_col], errors='coerce').dropna()
    if x_vals_all.empty:
        return None
    x_lo = float(x_vals_all.min())
    x_hi = float(x_vals_all.max())
    x_pad = max(0.05 * (x_hi - x_lo), 1e-3)
    x_grid = np.linspace(max(0.0, x_lo - x_pad), x_hi + x_pad, 200)

    series_summary = []  # for the legend annotation / caption

    for i, mid in enumerate(selected_method_ids):
        col_color = palette[i % len(palette)]
        if mid.startswith('ml::'):
            model_name = mid[4:]
            sub = bench_df[bench_df.get('Model', '').astype(str) == model_name] \
                  if 'Model' in bench_df.columns else bench_df
            if sub.empty:
                continue
            xs = pd.to_numeric(sub[x_col], errors='coerce')
            ys = pd.to_numeric(sub.get('Predicted_km'), errors='coerce')
            label = f'ML · {model_name}'
            method_col = 'Predicted_km'
            formula_text = None
        elif mid.startswith('proxy::'):
            method_col = mid[7:]
            if method_col not in bench_df.columns:
                continue
            xs = pd.to_numeric(bench_df[x_col], errors='coerce')
            ys = pd.to_numeric(bench_df[method_col], errors='coerce')
            label = PROXY_THICKNESS_LABELS.get(method_col, method_col)
            formula_text = PROXY_THICKNESS_FORMULAS.get(method_col)
        else:
            continue

        ok = xs.notna() & ys.notna()
        if not ok.any():
            continue
        xs_ok = xs[ok].to_numpy()
        ys_ok = ys[ok].to_numpy()

        # Scatter trace (samples)
        fig.add_trace(go.Scattergl(
            x=xs_ok, y=ys_ok, mode='markers',
            name=label,
            marker=dict(size=point_size, color=col_color,
                        line=dict(color='rgba(17,24,39,0.45)', width=0.4)),
            hovertemplate=(f'<b>{label}</b><br>'
                           f'{x_col}=%{{x:.2f}}<br>H=%{{y:.1f}} km'
                           '<extra></extra>'),
            legendgroup=label,
        ))

        # Formula curve overlay (only for proxy methods where x_col matches the formula's input)
        if show_curves and mid.startswith('proxy::'):
            curve_y = _unified_formula_curve(method_col, x_col, x_grid)
            if curve_y is not None:
                ok_curve = np.isfinite(curve_y)
                if ok_curve.any():
                    fig.add_trace(go.Scatter(
                        x=x_grid[ok_curve], y=curve_y[ok_curve], mode='lines',
                        name=f'{label} curve',
                        line=dict(color=col_color, width=2.0, dash='dot'),
                        legendgroup=label,
                        hovertemplate=(f'<b>{label} formula</b><br>'
                                       f'{x_col}=%{{x:.2f}} → H=%{{y:.1f}} km'
                                       '<extra></extra>'),
                    ))

        # Per-series median (horizontal line)
        if show_medians and ys_ok.size > 0:
            med = float(np.median(ys_ok))
            fig.add_hline(y=med, line=dict(color=col_color, width=1.2, dash='dash'),
                          opacity=0.55,
                          annotation_text=f'{label} median = {med:.1f} km',
                          annotation_position='top right',
                          annotation_font=dict(size=9, color=col_color))

        # Best-fit line per series (least squares, scoped to series X range)
        if show_best_fit and xs_ok.size >= 2:
            try:
                slope, intercept = np.polyfit(xs_ok, ys_ok, 1)
                xs_fit = np.linspace(xs_ok.min(), xs_ok.max(), 50)
                ys_fit = slope * xs_fit + intercept
                fig.add_trace(go.Scatter(
                    x=xs_fit, y=ys_fit, mode='lines',
                    name=f'{label} best fit',
                    line=dict(color=col_color, width=1.4),
                    legendgroup=label, showlegend=False,
                    hovertemplate=(f'<b>{label} best fit</b><br>'
                                   f'slope={slope:.2f}, intercept={intercept:.2f}'
                                   '<extra></extra>'),
                ))
            except Exception:
                pass

        series_summary.append({
            'Series':  label,
            'N':       int(ok.sum()),
            'Median':  float(np.median(ys_ok)) if ys_ok.size else float('nan'),
            'Formula': formula_text or '',
        })

    # 1:1 reference line — only meaningful when both axes are thickness-scale
    if show_one_to_one and x_col in ('Observed_km', 'Predicted_km'):
        _max_axis = max(x_hi, max((np.nanmax(t.y) for t in fig.data
                                    if t.y is not None and len(t.y) > 0), default=0))
        fig.add_trace(go.Scatter(
            x=[0, _max_axis], y=[0, _max_axis], mode='lines',
            name='1:1', line=dict(color='black', dash='dash', width=1.2),
            hoverinfo='skip', showlegend=True,
        ))

    fig.update_layout(
        height=height, template='plotly_white',
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=x_col, yaxis_title='crustal thickness [km]',
        legend=dict(orientation='h', y=-0.18, x=0,
                    font=dict(size=9), bgcolor='rgba(255,255,255,0.6)'),
    )
    return fig, pd.DataFrame(series_summary)
    # _LEGACY_unified_orphan_END — anything below this point in the legacy
    # block is dead code wrapped inside the marker function and never runs.


# Hierarchy hints printed next to the geologic-time columns in every
# grouping dropdown. Era ⊃ Period ⊃ Epoch ⊃ Age_Label; without these
# annotations users see four sibling-looking options and have no clue
# they're related at different resolutions.
_GEOLOGIC_TIME_HIERARCHY_HINT = {
    'Geologic_Era':       '(coarse — eras)',
    'Geologic_Period':    '(periods)',
    'Geologic_Epoch':     '(epochs)',
    'Geologic_Age_Label': '(fine — ages)',
    'Age_Ma':             '(numeric, Ma)',
}


def group_picker_label(c):
    """Decorate column names in grouping dropdowns with a hierarchy hint
    when the column is one of the geologic-time labels. Used as a
    ``format_func`` on every "Group rows by" / "And also by" picker so
    the Era→Period→Epoch→Age_Label hierarchy reads at a glance."""
    s = str(c)
    hint = _GEOLOGIC_TIME_HIERARCHY_HINT.get(s)
    return f'{s}  {hint}' if hint else s


def simple_xy_two_stage_picker(label, axis_options, default_col,
                               key_prefix, container=None,
                               include_none=False, none_label='(none)'):
    """Render a two-stage column picker: category dropdown → column dropdown.

    Stage 1 lists every category that has at least one column on the
    current bench (sorted by ``_SXY_CATEGORY_INDEX``). Stage 2 lists
    every column inside that category. Picking a different category
    reuses the previous column when it's still in scope, otherwise it
    snaps to the first column in the new category.

    Returns the chosen column name (str), or ``None`` if the user picks
    the ``include_none`` sentinel.

    Parameters
    ----------
    label : str
        Caption rendered above the pair (e.g. "X axis").
    axis_options : list[str]
        Columns the picker can offer — caller has already filtered to
        usable bench columns.
    default_col : str
        Initial column on first render. Drives the initial category too.
    key_prefix : str
        Streamlit widget-key namespace; needed because we render multiple
        pickers per page (X, Y, Colour) and they must not collide.
    container : streamlit container, optional
        Render inside ``container`` (e.g. one of ``st.columns`` outputs).
        Defaults to the current ``st`` namespace.
    include_none : bool
        Add a "(none)" sentinel as the first category — handy for a
        Colour selector that defaults to "no colour mapping".
    none_label : str
        Label to show for the sentinel.
    """
    target = container if container is not None else st

    # Build {category → [cols]} but only include categories that
    # actually have columns. Sort by the canonical category index so
    # the dropdown order is consistent across the app.
    cat_map: dict = {}
    for c in axis_options:
        cat_map.setdefault(simple_xy_column_category(c), []).append(c)
    cats = sorted(cat_map.keys(),
                  key=lambda k: _SXY_CATEGORY_INDEX.get(k, 99))

    if not cats:
        target.info(f'{label}: no usable columns.')
        return None

    cat_choices = ([none_label] + cats) if include_none else cats

    # Decide the initial category: prefer the one matching default_col
    if include_none and default_col is None:
        default_cat = none_label
    else:
        default_cat = simple_xy_column_category(default_col) if default_col in axis_options \
                      else cats[0]
        if default_cat not in cat_choices:
            default_cat = cat_choices[0]

    # Persist the user's category pick so the column dropdown stays
    # locked to it across reruns.
    cat_key = f'{key_prefix}__cat'
    col_key = f'{key_prefix}__col'

    target.markdown(f'**{label}**')
    chosen_cat = target.selectbox(
        f'{label} — group',
        cat_choices,
        index=cat_choices.index(st.session_state.get(cat_key, default_cat))
              if st.session_state.get(cat_key, default_cat) in cat_choices else 0,
        key=cat_key,
        label_visibility='collapsed',
    )

    if include_none and chosen_cat == none_label:
        return None

    cols_in_cat = cat_map.get(chosen_cat, [])
    if not cols_in_cat:
        return None

    # Resolve the default column inside the chosen category. Order:
    #   1. The previously-stored col_key value, if it's still in scope.
    #   2. The default_col, if it lives in this category.
    #   3. The first column of the category.
    prev = st.session_state.get(col_key)
    if prev in cols_in_cat:
        col_default = prev
    elif default_col in cols_in_cat:
        col_default = default_col
    else:
        col_default = cols_in_cat[0]

    chosen_col = target.selectbox(
        f'{label} — column',
        cols_in_cat,
        index=cols_in_cat.index(col_default),
        key=col_key,
        label_visibility='collapsed',
    )
    return chosen_col


def composite_group_column(bench_df, primary_col, secondary_col=None,
                           age_col='Age_Ma', age_bin_ma=None,
                           dest_col='_Composite_Group'):
    """Compose a synthetic group column from one or two source columns.

    Real-world grouping in geology often needs space × time (e.g. *"Andes
    Cretaceous"* vs *"Andes Neogene"* — the same arc has different
    crustal architecture at different ages). This helper returns
    ``(bench_with_composite, composite_col_name)`` where ``composite_col_name``
    is either:

      • ``primary_col`` — when no secondary is given.
      • ``dest_col``     — a new column in the returned DataFrame that
        concatenates ``primary_col`` and the secondary (categorical or
        Age-bin) with a ``" · "`` separator. Examples:
        ``"Andes · Neogene"``, ``"Aleutians · 0–10 Ma"``.

    Age-bin path is triggered when ``age_bin_ma`` is a positive number
    and ``age_col`` exists on ``bench_df``. Bins are right-open at the
    upper edge (``[lo, hi)``) and labelled in Ma. Rows with NaN ``age_col``
    fall into a ``"unknown age"`` bucket so they're still groupable.
    """
    if bench_df is None or bench_df.empty or not primary_col:
        return bench_df, primary_col
    if primary_col not in bench_df.columns:
        return bench_df, primary_col

    # No secondary requested — return as-is.
    if not secondary_col and not age_bin_ma:
        return bench_df, primary_col

    out = bench_df.copy()
    primary_str = out[primary_col].astype(str).fillna('—')

    # Secondary: either a categorical column or Age bins
    if age_bin_ma and age_col in out.columns:
        ages = pd.to_numeric(out[age_col], errors='coerce')
        # Build right-open bins from 0 up to ceil(max age / bin) * bin
        if ages.notna().any():
            _hi = float(ages.max())
            _lo = float(ages.min())
            _step = float(age_bin_ma)
            _start = max(0.0, np.floor(_lo / _step) * _step)
            _stop  = (np.ceil(_hi / _step) + 1) * _step
            _edges = np.arange(_start, _stop, _step)
            if len(_edges) < 2:
                _edges = np.array([_start, _start + _step])
            _labels = [f'{_edges[i]:.0f}–{_edges[i+1]:.0f} Ma'
                       for i in range(len(_edges) - 1)]
            _binned = pd.cut(ages, bins=_edges, labels=_labels,
                             right=False, include_lowest=True)
            secondary_str = _binned.astype(str).where(_binned.notna(), 'unknown age')
        else:
            secondary_str = pd.Series(['unknown age'] * len(out), index=out.index)
    elif secondary_col and secondary_col in out.columns:
        secondary_str = out[secondary_col].astype(str).fillna('—')
    else:
        return bench_df, primary_col

    out[dest_col] = primary_str + ' · ' + secondary_str
    return out, dest_col


def render_game_settings_card(key_suffix=''):
    """Render a small read-only card showing the GAME (Luffi & Ducea 2022)
    parameters that the unified method graph honours.

    The values shown reflect the constants used by ``apply_luffi_game_mohometers``
    (alpha, beta) and the default GAME filtering thresholds. We render
    them disabled here so users on Validate / Predict can see what was
    used to compute the GAME consensus column without being able to
    nudge it inline — the calibration is set on the Model tab and
    propagated through ``enrich(..., include_game=True)``.
    """
    st.caption('GAME (Luffi & Ducea 2022) consensus parameters — preset from the Model tab, read-only here.')
    _gc1, _gc2, _gc3, _gc4, _gc5 = st.columns(5)
    _gc1.metric('α (slope)',     f'{GAME_ALPHA_DEFAULT:.2f}',
                help='Elevation → Moho slope. H = α·elev + β')
    _gc2.metric('β (intercept)', f'{GAME_BETA_DEFAULT:.2f} km')
    _gc3.metric('focus MAD',     '5.0 km',
                help='Filter threshold: drop any sample whose mohometer MAD exceeds this in the focus pass')
    _gc4.metric('final MAD',     '10.0 km',
                help='Filter threshold: final MAD ceiling after the focus pass')
    _gc5.metric('min mohometers','3',
                help='Minimum surviving mohometer count required to report a GAME consensus value')


def local_option_label(c):
    if str(c).startswith('Local_Median_'):
        base = str(c).replace('Local_Median_','')
        return 'local median: ' + local_option_label(base)
    labels = {
        'Predicted_km':'model: crustal thickness [Km]',
        'Observed_km':'known: crustal thickness [Km]',
        'Residual_km':'model - known [Km]',
        'Age_Ma':'Age [Ma]',
        'Sr_Y':'Sr/Y',
        'La_Yb_N':'La/Yb(N)',
        'Ce_Y':'Ce/Y',
        'La_Y':'La/Y',
        'Rb_Sr':'Rb/Sr',
        'Dy_Yb':'Dy/Yb',
        'Gd_Yb':'Gd/Yb',
        'Zr_Ti':'Zr/Ti',
        'Elevation_km':'Elevation [km]',
        'MgO':'MgO',
        'SiO2':'SiO2',
        'Rock_Type_Model':'Rock type',
        'Geologic_Domain':'Geological domain',
        'Arc_or_Segment':'Arc/segment',
        'Dataset':'Dataset',
        'Grouping_Method':'Grouping method',
        'Local_N':'Local N',
        'Local_Domain_Value':'Group ID',
        'GAME_Luffi2022_N_mohometers':'GAME mohometers kept',
        'GAME_Luffi2022_N_raw_mohometers':'GAME mohometers raw',
        'GAME_Luffi2022_MAD_km':'GAME MAD [Km]',
        'GAME_Luffi2022_IQR_km':'GAME IQR [Km]',
        'GAME_Luffi2022_CI95_Width_km':'GAME 95% CI width [Km]',
        'GAME_Luffi2022_Reliability':'GAME reliability',
    }
    return labels.get(c, PROXY_THICKNESS_LABELS.get(c, str(c).replace('_',' ')))

def model_thickness_label(model_name):
    if not model_name or str(model_name).strip() in ('','nan'):
        return 'model: crustal thickness [Km]'
    return f'{model_name}: crustal thickness [Km]'

def validation_proxy_thickness_figure(df, x_col, methods, point_size=5, show_formulas=False, trend_stat='Median', trend_bin_width=5.0, trend_min_n=5, x_range=None, y_range=None):
    if df.empty or x_col not in df or not methods:
        return None, {}
    df = plot_df(df)
    model_suffix = ''
    if x_col == 'Predicted_km' and 'Model' in df and df['Model'].nunique(dropna=True) == 1:
        model_suffix = f' ({df["Model"].dropna().astype(str).iloc[0]})'
    if x_col == 'Observed_km':
        x_label = 'known: crustal thickness [Km]'
    elif x_col == 'Predicted_km':
        x_label = f'model: crustal thickness{model_suffix} [Km]'
    else:
        x_label = PROXY_THICKNESS_LABELS.get(x_col, x_col) + ' [Km]'
    game_cols = [c for c in ['GAME_Luffi2022_N_mohometers','GAME_Luffi2022_N_raw_mohometers','GAME_Luffi2022_MAD_km','GAME_Luffi2022_Status'] if c in df]
    id_cols = [c for c in ['Sample_ID','Lat','Lon','Model',x_col] if c in df]
    cols = list(dict.fromkeys(id_cols + game_cols + methods))
    plot = df[cols].copy()
    if x_col == 'Observed_km':
        dedupe = [c for c in ['Sample_ID','Lat','Lon',x_col] if c in plot]
        plot = plot.drop_duplicates(subset=dedupe if dedupe else None)
    long = plot.melt(id_vars=list(dict.fromkeys(id_cols + game_cols)), value_vars=[m for m in methods if m in plot], var_name='Proxy_Method', value_name='Proxy_Thickness_km')
    long['Method'] = long['Proxy_Method'].map(PROXY_THICKNESS_LABELS).fillna(long['Proxy_Method'])
    long[x_col] = pd.to_numeric(long[x_col], errors='coerce')
    long['Proxy_Thickness_km'] = pd.to_numeric(long['Proxy_Thickness_km'], errors='coerce')
    long = long.dropna(subset=[x_col,'Proxy_Thickness_km'])
    if long.empty:
        return None, {}
    long['Delta_km'] = long['Proxy_Thickness_km'] - long[x_col]
    game_n = pd.to_numeric(long['GAME_Luffi2022_N_mohometers'], errors='coerce') if 'GAME_Luffi2022_N_mohometers' in long else pd.Series(np.nan, index=long.index)
    game_raw = pd.to_numeric(long['GAME_Luffi2022_N_raw_mohometers'], errors='coerce') if 'GAME_Luffi2022_N_raw_mohometers' in long else pd.Series(np.nan, index=long.index)
    game_mad = pd.to_numeric(long['GAME_Luffi2022_MAD_km'], errors='coerce') if 'GAME_Luffi2022_MAD_km' in long else pd.Series(np.nan, index=long.index)
    game_status = long['GAME_Luffi2022_Status'].fillna('').astype(str) if 'GAME_Luffi2022_Status' in long else pd.Series('', index=long.index)
    game_n_text = game_n.round(0).astype('Int64').astype(str) + ' GAME'
    long['GAME_N_Display'] = np.where(long['Proxy_Method'].eq('H_GAME_LuffiDucea2022_km') & game_n.notna(), game_n_text, '')
    long['GAME_Info'] = np.where(
        long['Proxy_Method'].eq('H_GAME_LuffiDucea2022_km') & game_n.notna(),
        'kept=' + game_n.round(0).astype('Int64').astype(str) +
        np.where(game_raw.notna(), '; raw=' + game_raw.round(0).astype('Int64').astype(str), '') +
        np.where(game_mad.notna(), '; MAD=' + game_mad.round(1).astype(str) + ' Km', '') +
        np.where(game_status.ne(''), '; ' + game_status, ''),
        ''
    )
    if x_col == 'Observed_km':
        delta_label = 'proxy - known [Km]'
    elif x_col == 'Predicted_km':
        delta_label = 'proxy - model [Km]'
    else:
        delta_label = f'proxy - {PROXY_THICKNESS_LABELS.get(x_col, x_col)} [Km]'
    fig = px.scatter(
        plot_df(tidy_numbers(long)), x=x_col, y='Proxy_Thickness_km', color='Delta_km', symbol='Method',
        color_continuous_scale=RESIDUAL_COLORSCALE, color_continuous_midpoint=0, template='plotly_white',
        labels={x_col:x_label,'Proxy_Thickness_km':'proxy: crustal thickness [Km]','Method':'Proxy method','Delta_km':delta_label},
        custom_data=['Delta_km','Method','GAME_Info']
    )
    fig.update_traces(marker=dict(size=point_size,line=dict(color='black',width=0.6)),hovertemplate=x_label+'=%{x:.1f}<br>proxy: crustal thickness=%{y:.1f} Km<br>Delta=%{customdata[0]:+.1f} Km<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>')
    fig.add_trace(go.Scatter(x=[0,90],y=[0,90],mode='lines',name='1:1',line=dict(color='black',dash='dash')))
    fig = add_binned_summary_traces(fig, long, x_col, 'Proxy_Thickness_km', trend_stat, trend_bin_width, trend_min_n, 'Method', 'moving')
    fig.update_layout(template='plotly_white',height=460,margin=dict(l=20,r=20,t=20,b=20),legend=dict(orientation='h',y=-0.24,x=0))
    fig.update_xaxes(range=x_range if x_range else None, showgrid=True)
    fig.update_yaxes(range=y_range if y_range else None, showgrid=True)
    formulas_out = {PROXY_THICKNESS_LABELS.get(m,m):PROXY_THICKNESS_FORMULAS[m] for m in methods if m in PROXY_THICKNESS_FORMULAS} if show_formulas else {}
    return fig, formulas_out

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def nearest_crust_matches(samples, grid):
    required = {'Lat','Lon'}
    if not required.issubset(samples) or not required.issubset(grid):
        raise ValueError('Both geochemistry samples and CRUST1.0 grid need Lat and Lon columns.')
    crust_cols = [c for c in ['CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Vp','CRUST1_Density'] if c in grid]
    if 'CRUST1_Total_Crust_km' not in crust_cols:
        raise ValueError('CRUST1.0 grid needs a total crust thickness column.')
    sample = samples.dropna(subset=['Lat','Lon']).copy()
    crust = grid.dropna(subset=['Lat','Lon','CRUST1_Total_Crust_km']).copy()
    if sample.empty or crust.empty:
        raise ValueError('No matchable sample/grid rows remain after dropping missing Lat, Lon, or total crust thickness.')
    glat = crust['Lat'].to_numpy(dtype=float)
    glon = crust['Lon'].to_numpy(dtype=float)
    if cKDTree is not None:
        tree = cKDTree(np.column_stack([glat, glon]))
        slat = sample['Lat'].to_numpy(dtype=float)
        slon = sample['Lon'].to_numpy(dtype=float)
        _, idxs = tree.query(np.column_stack([slat, slon]))
        nearest_idx = [crust.index[int(i)] for i in idxs]
        nearest_dist = [float(haversine_km(slat[i], slon[i], glat[idxs[i]], glon[idxs[i]])) for i in range(len(idxs))]
    else:
        # Vectorised fallback when scipy is unavailable: broadcast haversine over sample × grid
        slat = sample['Lat'].to_numpy(dtype=float)
        slon = sample['Lon'].to_numpy(dtype=float)
        # shape (n_samples, n_grid) via broadcasting
        dlat = np.radians(slat[:, None] - glat[None, :])
        dlon = np.radians(slon[:, None] - glon[None, :])
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(slat[:, None])) * np.cos(np.radians(glat[None, :])) * np.sin(dlon / 2) ** 2
        dist_matrix = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        idxs = np.nanargmin(dist_matrix, axis=1)
        nearest_idx = [crust.index[int(i)] for i in idxs]
        nearest_dist = [float(dist_matrix[r, idxs[r]]) for r in range(len(idxs))]
    matched_grid = crust.loc[nearest_idx, ['Lat','Lon'] + crust_cols].reset_index(drop=True).add_prefix('Nearest_')
    out = sample.reset_index(drop=True).join(matched_grid)
    out['CRUST1_Match_Distance_km'] = nearest_dist
    out['CRUST1_Benchmark_km'] = out['Nearest_CRUST1_Total_Crust_km']
    if 'H_Guo_ERT_km' in out:
        out['Residual_Guo_vs_CRUST1_km'] = out['H_Guo_ERT_km'] - out['CRUST1_Benchmark_km']
    if 'H_Sundell2021_Paired_km' in out:
        out['Residual_Sundell_vs_CRUST1_km'] = out['H_Sundell2021_Paired_km'] - out['CRUST1_Benchmark_km']
    zou = 'H_Zou2021_LaYbN_SVRE_km' if 'H_Zou2021_LaYbN_SVRE_km' in out else 'H_Zou2021_SrY_SVRE_km'
    if zou in out:
        out['Residual_Zou_vs_CRUST1_km'] = out[zou] - out['CRUST1_Benchmark_km']
    return out

@st.cache_data(show_spinner=False)
def attach_crust1_reference(df):
    """Attach CRUST1.0 and LithoRef18 reference values (triangulated linear interpolation) to a benchmarked dataframe."""
    out = df.copy()
    if out.empty or not {'Lat','Lon'}.issubset(out):
        return out
    coords = out[['Lat','Lon']].apply(pd.to_numeric, errors='coerce')
    ok = coords['Lat'].notna() & coords['Lon'].notna()
    if not ok.any():
        return out
    slat = coords.loc[ok,'Lat'].to_numpy()
    slon = coords.loc[ok,'Lon'].to_numpy()
    # CRUST1.0
    crust1 = read_default_crust_grid(str(DEFAULT_CRUST1_GRID))
    if not crust1.empty and 'CRUST1_Total_Crust_km' in crust1:
        out['CRUST1_Total_Crust_km'] = np.nan
        out.loc[ok,'CRUST1_Total_Crust_km'] = grid_interpolate(slat, slon, crust1, 'CRUST1_Total_Crust_km', 'CRUST1.0')
        if 'CRUST1_Crystalline_Crust_km' in crust1:
            out['CRUST1_Crystalline_Crust_km'] = np.nan
            out.loc[ok,'CRUST1_Crystalline_Crust_km'] = grid_interpolate(slat, slon, crust1, 'CRUST1_Crystalline_Crust_km', 'CRUST1.0_cryst')
        if {'CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km'}.issubset(out):
            out['CRUST1_Sediment_km'] = out['CRUST1_Total_Crust_km'] - out['CRUST1_Crystalline_Crust_km']
        out['CRUST1_Benchmark_km'] = out['CRUST1_Total_Crust_km']
    # LithoRef18 (Alfonso 2019)
    lithoref = read_default_lithoref18()
    if not lithoref.empty and 'LithoRef18_Total_Crust_km' in lithoref:
        out['LithoRef18_Total_Crust_km'] = np.nan
        out.loc[ok,'LithoRef18_Total_Crust_km'] = grid_interpolate(slat, slon, lithoref, 'LithoRef18_Total_Crust_km', 'LithoRef18')
        if 'LithoRef18_LAB_km' in lithoref:
            out['LithoRef18_LAB_km'] = np.nan
            out.loc[ok,'LithoRef18_LAB_km'] = grid_interpolate(slat, slon, lithoref, 'LithoRef18_LAB_km', 'LithoRef18_LAB')
    return out

def bootstrap_median_ci(values, n_boot=500, seed=42):
    vals = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(vals, size=(int(n_boot), len(vals)), replace=True), axis=1)
    return float(np.nanpercentile(meds, 2.5)), float(np.nanpercentile(meds, 97.5))

def neighbourhood_stats(values, n_boot=500, seed=42):
    vals = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    if vals.empty:
        return {
            'Local_N':0,'Local_Median':np.nan,'Local_Mean':np.nan,'Local_SD':np.nan,
            'Local_Q25':np.nan,'Local_Q75':np.nan,'Local_IQR':np.nan,'Local_Min':np.nan,'Local_Max':np.nan,
            'Bootstrap_95CI_Low_Median':np.nan,'Bootstrap_95CI_High_Median':np.nan,
        }
    low, high = bootstrap_median_ci(vals, n_boot=n_boot, seed=seed)
    q25 = float(vals.quantile(0.25)); q75 = float(vals.quantile(0.75))
    return {
        'Local_N':int(len(vals)),
        'Local_Median':float(vals.median()),
        'Local_Mean':float(vals.mean()),
        'Local_SD':float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        'Local_Q25':q25,
        'Local_Q75':q75,
        'Local_IQR':float(q75-q25),
        'Local_Min':float(vals.min()),
        'Local_Max':float(vals.max()),
        'Bootstrap_95CI_Low_Median':low,
        'Bootstrap_95CI_High_Median':high,
    }

def spatiotemporal_local_estimates(targets, samples, value_col, mode='Modern benchmark', target_age_col='Age_Ma',
                                  time_window=10.0, initial_radius_km=100.0, max_radius_km=250.0, radius_step_km=50.0,
                                  minimum_n=10, preferred_n=30, fallback='nearest N', n_boot=500, seed=42):
    if targets.empty or samples.empty or value_col not in samples or not {'Lat','Lon'}.issubset(targets) or not {'Lat','Lon'}.issubset(samples):
        return pd.DataFrame()
    candidates = samples.copy()
    for c in ['Lat','Lon','Age_Ma',value_col]:
        if c in candidates:
            candidates[c] = pd.to_numeric(candidates[c], errors='coerce')
    candidates = candidates.dropna(subset=['Lat','Lon',value_col]).reset_index(drop=True)
    if candidates.empty:
        return pd.DataFrame()
    rows = []
    model_groups = list(candidates.groupby('Model', dropna=False)) if 'Model' in candidates else [('all', candidates)]
    for model_name, model_work in model_groups:
      for i, target in targets.reset_index(drop=True).iterrows():
        if 'Model' in target.index and pd.notna(target.get('Model')) and str(target.get('Model')) != str(model_name):
            continue
        tlat = pd.to_numeric(pd.Series([target.get('Lat')]), errors='coerce').iloc[0]
        tlon = pd.to_numeric(pd.Series([target.get('Lon')]), errors='coerce').iloc[0]
        if pd.isna(tlat) or pd.isna(tlon):
            continue
        pool = candidates.copy()
        used_time_window = np.nan
        if mode == 'Modern benchmark' and 'Age_Ma' in pool:
            pool = pool[pool['Age_Ma'] <= 5]
            used_time_window = 5.0
        elif mode == 'Time-slice' and 'Age_Ma' in pool:
            target_age = pd.to_numeric(pd.Series([target.get(target_age_col, target.get('Age_Ma', np.nan))]), errors='coerce').iloc[0]
            if pd.notna(target_age):
                used_time_window = float(time_window)
                pool = pool[pool['Age_Ma'].sub(float(target_age)).abs() <= used_time_window / 2]
        if pool.empty:
            selected = pool.copy()
            used_radius = np.nan
            fallback_used = 'no candidates after time filter'
        else:
            pool = pool.copy()
            pool['Local_Distance_km'] = haversine_km(float(tlat), float(tlon), pool['Lat'].to_numpy(dtype=float), pool['Lon'].to_numpy(dtype=float))
            selected = pd.DataFrame()
            used_radius = float(initial_radius_km)
            fallback_used = ''
            radius = float(initial_radius_km)
            while radius <= float(max_radius_km) + 1e-9:
                selected = pool[pool['Local_Distance_km'] <= radius].copy()
                used_radius = radius
                if len(selected) >= int(minimum_n):
                    break
                radius += float(radius_step_km)
            if len(selected) < int(minimum_n):
                if fallback == 'nearest N':
                    selected = pool.sort_values('Local_Distance_km').head(int(preferred_n)).copy()
                    fallback_used = 'nearest N'
                    used_radius = float(selected['Local_Distance_km'].max()) if not selected.empty else np.nan
                elif fallback == 'expand time window' and mode == 'Time-slice' and 'Age_Ma' in candidates:
                    target_age = pd.to_numeric(pd.Series([target.get(target_age_col, target.get('Age_Ma', np.nan))]), errors='coerce').iloc[0]
                    expanded = candidates.copy()
                    if pd.notna(target_age):
                        used_time_window = float(time_window) * 2
                        expanded = expanded[expanded['Age_Ma'].sub(float(target_age)).abs() <= used_time_window / 2].copy()
                    if not expanded.empty:
                        expanded['Local_Distance_km'] = haversine_km(float(tlat), float(tlon), expanded['Lat'].to_numpy(dtype=float), expanded['Lon'].to_numpy(dtype=float))
                        selected = expanded[expanded['Local_Distance_km'] <= float(max_radius_km)].copy()
                    fallback_used = 'expand time window'
                    used_radius = float(max_radius_km)
                else:
                    selected = pd.DataFrame()
                    fallback_used = 'no estimate'
        stats = neighbourhood_stats(selected[value_col] if value_col in selected else [], n_boot=n_boot, seed=seed+i)
        row = {c:target.get(c) for c in targets.columns}
        row.update(stats)
        row.update({
            'Local_Value_Column':value_col,
            'Local_Mode':mode,
            'Local_Radius_km':used_radius,
            'Local_Time_Window_Ma':used_time_window,
            'Local_Fallback':fallback_used,
        })
        rows.append(row)
    return tidy_numbers(pd.DataFrame(rows))

def lonlat_xy_km(lon, lat, ref_lat=None):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    ref = np.nanmean(lat) if ref_lat is None else float(ref_lat)
    x = lon * 111.32 * np.cos(np.radians(ref))
    y = lat * 110.57
    return np.column_stack([x,y])

def long_axis_km_from_lonlat(lon, lat):
    pts = lonlat_xy_km(lon, lat)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return np.nan
    centered = pts - pts.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = centered @ vh[0]
        return float(np.nanmax(axis) - np.nanmin(axis))
    except Exception:
        d = np.sqrt(((pts[:,None,:] - pts[None,:,:]) ** 2).sum(axis=2))
        return float(np.nanmax(d))

def polygon_contains_lonlat(poly_lonlat, lon, lat):
    poly = np.asarray(poly_lonlat, dtype=float)
    if len(poly) < 3:
        return False
    return bool(MplPath(poly).contains_point((float(lon), float(lat))))

def geojson_domain_records(uploaded):
    obj = json.loads(uploaded.getvalue().decode('utf-8'))
    records = []
    for i, feat in enumerate(obj.get('features', [])):
        props = feat.get('properties', {}) or {}
        geom = feat.get('geometry', {}) or {}
        domain = props.get('Domain') or props.get('domain') or props.get('Name') or props.get('name') or props.get('id') or f'Domain {i+1}'
        coords = geom.get('coordinates', [])
        polys = []
        if geom.get('type') == 'Polygon':
            polys = [coords]
        elif geom.get('type') == 'MultiPolygon':
            polys = [p for p in coords]
        for j, poly in enumerate(polys):
            if not poly:
                continue
            ring = np.asarray(poly[0], dtype=float)
            if ring.ndim != 2 or ring.shape[1] < 2:
                continue
            lon = ring[:,0]
            lat = ring[:,1]
            records.append({
                'Domain':str(domain if len(polys) == 1 else f'{domain} {j+1}'),
                'Polygon':np.column_stack([lon,lat]),
                'Long_Axis_km':long_axis_km_from_lonlat(lon,lat),
            })
    return records

def table_domain_records(uploaded):
    df = read_table(uploaded)
    domain_col = next((c for c in ['Domain','Geological_Domain','Arc_or_Segment','Dataset','Tectonic_Setting'] if c in df), None)
    if domain_col is None or not {'Lat','Lon'}.issubset(df):
        return []
    records = []
    for domain, g in df.dropna(subset=['Lat','Lon']).groupby(domain_col):
        if len(g) < 3:
            continue
        lon = pd.to_numeric(g['Lon'], errors='coerce')
        lat = pd.to_numeric(g['Lat'], errors='coerce')
        ok = lon.notna() & lat.notna()
        if ok.sum() < 3:
            continue
        records.append({
            'Domain':str(domain),
            'Polygon':np.column_stack([lon[ok].to_numpy(dtype=float),lat[ok].to_numpy(dtype=float)]),
            'Long_Axis_km':long_axis_km_from_lonlat(lon[ok],lat[ok]),
        })
    return records

def read_domain_records(uploaded):
    if uploaded is None:
        return []
    name = getattr(uploaded,'name','').lower()
    if name.endswith(('.geojson','.json')):
        return geojson_domain_records(uploaded)
    return table_domain_records(uploaded)

def attach_uploaded_geological_domains(df, records):
    out = df.copy()
    out['Geological_Domain'] = np.nan
    out['Geological_Domain_Long_Axis_km'] = np.nan
    if not records or not {'Lat','Lon'}.issubset(out):
        return out
    for idx, row in out.dropna(subset=['Lat','Lon']).iterrows():
        for rec in records:
            if polygon_contains_lonlat(rec['Polygon'], row['Lon'], row['Lat']):
                out.at[idx,'Geological_Domain'] = rec['Domain']
                out.at[idx,'Geological_Domain_Long_Axis_km'] = rec['Long_Axis_km']
                break
    return out

@st.cache_data(show_spinner=False)
def attach_sample_distribution_domains(df, group_col=None):
    out = df.copy()
    out['Sample_Distribution_Domain'] = 'All samples'
    out['Sample_Distribution_Long_Axis_km'] = np.nan
    if not {'Lat','Lon'}.issubset(out):
        return out
    groups = [(None,out)] if not group_col or group_col not in out else out.groupby(group_col, dropna=False)
    for name, g in groups:
        g = g.dropna(subset=['Lat','Lon'])
        if g.empty:
            continue
        lon = pd.to_numeric(g['Lon'], errors='coerce')
        lat = pd.to_numeric(g['Lat'], errors='coerce')
        ok = lon.notna() & lat.notna()
        if ok.sum() < 2:
            continue
        long_axis = long_axis_km_from_lonlat(lon[ok], lat[ok])
        domain_name = 'All samples' if name is None else str(name)
        out.loc[g.index, 'Sample_Distribution_Domain'] = domain_name
        out.loc[g.index, 'Sample_Distribution_Long_Axis_km'] = long_axis
    return out

# auto-clustering removed — see engine/_parked_auto_clustering.py for the
# KMeans/DBSCAN/Agglomerative + GMM helpers and re-enable instructions.

def attach_long_axis_position(df, group_col=None, out_col='Long_Axis_Position_km'):
    out = df.copy()
    out[out_col] = np.nan
    out['Long_Axis_Length_km'] = np.nan
    out['Long_Axis_Orientation'] = np.nan
    if not {'Lat','Lon'}.issubset(out):
        return out
    groups = [(None,out)] if not group_col or group_col not in out else out.groupby(group_col, dropna=False)
    for _, g in groups:
        g = g.dropna(subset=['Lat','Lon'])
        if len(g) < 2:
            continue
        lon = pd.to_numeric(g['Lon'], errors='coerce')
        lat = pd.to_numeric(g['Lat'], errors='coerce')
        ok = lon.notna() & lat.notna()
        if ok.sum() < 2:
            continue
        pts = lonlat_xy_km(lon[ok], lat[ok])
        centered = pts - pts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            vec = vh[0].copy()
            axis = centered @ vec
            p_start = pts[np.nanargmin(axis)]
            p_end = pts[np.nanargmax(axis)]
            dx = p_end[0] - p_start[0]
            dy = p_end[1] - p_start[1]
            if (abs(dy) > abs(dx) * 0.35 and p_start[1] < p_end[1]) or (abs(dy) <= abs(dx) * 0.35 and p_start[0] > p_end[0]):
                vec *= -1
                axis = centered @ vec
                p_start = pts[np.nanargmin(axis)]
                p_end = pts[np.nanargmax(axis)]
                dx = p_end[0] - p_start[0]
                dy = p_end[1] - p_start[1]
            if abs(dy) <= abs(dx) * 0.35:
                start_label = 'W' if dx > 0 else 'E'
                end_label = 'E' if dx > 0 else 'W'
            elif abs(dx) <= abs(dy) * 0.35:
                start_label = 'S' if dy > 0 else 'N'
                end_label = 'N' if dy > 0 else 'S'
            else:
                start_label = ('S' if dy > 0 else 'N') + ('W' if dx > 0 else 'E')
                end_label = ('N' if dy > 0 else 'S') + ('E' if dx > 0 else 'W')
            length = float(np.nanmax(axis) - np.nanmin(axis))
            axis = axis - np.nanmin(axis)
            out.loc[g.index[ok], out_col] = axis
            out.loc[g.index[ok], 'Long_Axis_Length_km'] = length
            out.loc[g.index[ok], 'Long_Axis_Orientation'] = f'{start_label}-{end_label}'
        except Exception:
            continue
    return out

@st.cache_data(show_spinner=False)
def attach_preset_group_domains(df, group_col=None, max_segment_km=100.0, split_by_distance=True, age_bin_width=50.0):
    out = df.copy()
    out['Preset_Group_Source'] = group_col or 'All samples'
    out['Preset_Group_Base'] = 'All samples'
    if group_col and group_col != 'All samples' and group_col in out:
        if group_col == 'Age_Ma':
            ages = pd.to_numeric(out['Age_Ma'], errors='coerce')
            width = max(float(age_bin_width), 1.0)
            lo = np.floor(ages / width) * width
            hi = lo + width
            out['Preset_Group_Base'] = np.where(
                ages.notna(),
                [f'{a:.0f}-{b:.0f} Ma' if np.isfinite(a) and np.isfinite(b) else 'unknown age' for a,b in zip(lo,hi)],
                'unknown age'
            )
        else:
            out['Preset_Group_Base'] = out[group_col].fillna('unknown').astype(str)
    out = attach_long_axis_position(out, 'Preset_Group_Base', out_col='Preset_Group_Position_km')
    out['Preset_Group_Long_Axis_km'] = out.get('Long_Axis_Length_km', np.nan)
    out['Preset_Group_Orientation'] = out.get('Long_Axis_Orientation', np.nan)
    max_segment_km = max(float(max_segment_km), 1.0)
    if split_by_distance and {'Lat','Lon'}.issubset(out):
        position = pd.to_numeric(out['Preset_Group_Position_km'], errors='coerce')
        segment = np.floor(position / max_segment_km) + 1
        segment = pd.Series(segment, index=out.index).where(position.notna(), 1).fillna(1).astype(int)
        out['Preset_Group_Segment'] = segment
        out['Preset_Group_Segment_Label'] = [
            f'{(s-1)*max_segment_km:.0f}-{s*max_segment_km:.0f} km' for s in segment
        ]
        out['Preset_Group_ID'] = out['Preset_Group_Base'].astype(str) + ' / ' + out['Preset_Group_Segment_Label'].astype(str)
    else:
        out['Preset_Group_Segment'] = 1
        out['Preset_Group_Segment_Label'] = 'all'
        out['Preset_Group_ID'] = out['Preset_Group_Base'].astype(str)
    out['Preset_Group_Segment_Width_km'] = np.nan
    for group_id, g in out.groupby('Preset_Group_ID', dropna=False):
        if {'Lon','Lat'}.issubset(g) and len(g.dropna(subset=['Lon','Lat'])) >= 2:
            axis = long_axis_km_from_lonlat(g['Lon'], g['Lat'])
        else:
            axis = max_segment_km if split_by_distance else np.nan
        if split_by_distance and pd.notna(axis):
            axis = min(float(axis), max_segment_km)
        out.loc[g.index, 'Preset_Group_Segment_Width_km'] = axis
    return out

def preset_group_summary(df):
    if df.empty or 'Preset_Group_ID' not in df:
        return pd.DataFrame()
    rows = []
    for group_id, g in df.groupby('Preset_Group_ID', dropna=False):
        rows.append({
            'Grouping_Method':'Preset grouping',
            'Group_ID':group_id,
            'Source':g['Preset_Group_Source'].dropna().astype(str).iloc[0] if 'Preset_Group_Source' in g and not g['Preset_Group_Source'].dropna().empty else 'All samples',
            'n':len(g),
            'Long_Axis_km':long_axis_km_from_lonlat(g['Lon'],g['Lat']) if {'Lon','Lat'}.issubset(g) and len(g.dropna(subset=['Lon','Lat'])) >= 2 else np.nan,
            'Median_Age_Ma':pd.to_numeric(g['Age_Ma'],errors='coerce').median() if 'Age_Ma' in g else np.nan,
            'Median_H_km':pd.to_numeric(g['Predicted_km'],errors='coerce').median() if 'Predicted_km' in g else np.nan,
        })
    return tidy_numbers(pd.DataFrame(rows))

def validation_local_estimates(df, value_col='Predicted_km', age_window=10.0, initial_radius_km=100.0,
                               max_radius_km=250.0, radius_step_km=50.0, minimum_n=10,
                               rock_types=None, domain_col=None,
                               n_boot=500, seed=42, radius_col=None):
    if df.empty or value_col not in df or not {'Lat','Lon','Age_Ma','Model'}.issubset(df):
        return pd.DataFrame()
    rows = []
    work = df.copy().reset_index(drop=True)
    for c in ['Lat','Lon','Age_Ma',value_col,'Observed_km','Predicted_km']:
        if c in work:
            work[c] = pd.to_numeric(work[c], errors='coerce')
    work = work.dropna(subset=['Lat','Lon','Age_Ma',value_col])
    if rock_types and 'Rock_Type_Model' in work:
        work = work[work['Rock_Type_Model'].isin(rock_types)]
    for model_name, model_df in work.groupby('Model'):
        for idx, target in model_df.iterrows():
            pool = model_df.drop(index=idx, errors='ignore').copy()
            if domain_col and domain_col in pool.columns and domain_col in target.index and pd.notna(target.get(domain_col)):
                pool = pool[pool[domain_col].astype(str).eq(str(target.get(domain_col)))]
            pool = pool[pool['Age_Ma'].sub(target['Age_Ma']).abs() <= float(age_window)]
            if pool.empty:
                selected = pool
                used_radius = np.nan
                low_n = True
            else:
                pool['Local_Distance_km'] = haversine_km(float(target['Lat']), float(target['Lon']), pool['Lat'].to_numpy(dtype=float), pool['Lon'].to_numpy(dtype=float))
                local_max_radius = float(max_radius_km)
                if radius_col and radius_col in target.index and pd.notna(target.get(radius_col)):
                    local_max_radius = max(float(initial_radius_km), float(target.get(radius_col)))
                radius = min(float(initial_radius_km), local_max_radius)
                selected = pd.DataFrame()
                used_radius = radius
                while radius <= local_max_radius + 1e-9:
                    selected = pool[pool['Local_Distance_km'] <= radius]
                    used_radius = radius
                    if len(selected) >= int(minimum_n):
                        break
                    radius += float(radius_step_km)
                low_n = len(selected) < int(minimum_n)
            stats = neighbourhood_stats(selected[value_col] if value_col in selected else [], n_boot=n_boot, seed=seed+int(idx))
            row = target.to_dict()
            row.update(stats)
            row.update({
                'Local_Value_Column':value_col,
                'Local_Age_Window_Ma':float(age_window),
                'Local_Radius_km':used_radius,
                'Local_Minimum_N':int(minimum_n),
                'Local_Low_N':bool(low_n),
                'Local_Domain_Column':domain_col or 'none',
                'Local_Domain_Value':target.get(domain_col) if domain_col and domain_col in target.index else 'all',
                'Local_Radius_Source':radius_col or 'manual',
                'Local_Domain_Long_Axis_km':target.get(radius_col) if radius_col and radius_col in target.index else np.nan,
            })
            rows.append(row)
    return tidy_numbers(pd.DataFrame(rows))

def local_h_stats(values, n_boot=500, seed=42):
    stats = neighbourhood_stats(values, n_boot=n_boot, seed=seed)
    return {
        'Local_N':stats['Local_N'],
        'Local_Median_H':stats['Local_Median'],
        'Local_Median_H_km':stats['Local_Median'],
        'Local_Mean_H':stats['Local_Mean'],
        'Local_Mean_H_km':stats['Local_Mean'],
        'Local_Q25_H':stats['Local_Q25'],
        'Local_Q25_H_km':stats['Local_Q25'],
        'Local_Q75_H':stats['Local_Q75'],
        'Local_Q75_H_km':stats['Local_Q75'],
        'Local_IQR_H':stats['Local_IQR'],
        'Local_IQR_H_km':stats['Local_IQR'],
        'Local_SD_H':stats['Local_SD'],
        'Local_SD_H_km':stats['Local_SD'],
        'Local_Min_H':stats['Local_Min'],
        'Local_Min_H_km':stats['Local_Min'],
        'Local_Max_H':stats['Local_Max'],
        'Local_Max_H_km':stats['Local_Max'],
        'Bootstrap_95CI_Low_Median':stats['Bootstrap_95CI_Low_Median'],
        'Bootstrap_95CI_High_Median':stats['Bootstrap_95CI_High_Median'],
    }

@st.cache_data(show_spinner=False)
def build_local_targets(target_mode, samples, user_lat=None, user_lon=None, user_age=0.0, arc_col=None):
    if target_mode == 'User-defined point':
        return pd.DataFrame([{'Target_ID':'User point','Lat':user_lat,'Lon':user_lon,'Age_Ma':user_age,'Target_Type':'user-defined point'}])
    if target_mode == 'Arc segment' and arc_col and arc_col in samples:
        rows = []
        for name, g in samples.dropna(subset=['Lat','Lon']).groupby(arc_col, dropna=False):
            rows.append({
                'Target_ID':str(name),
                'Lat':pd.to_numeric(g['Lat'], errors='coerce').median(),
                'Lon':pd.to_numeric(g['Lon'], errors='coerce').median(),
                'Age_Ma':pd.to_numeric(g['Age_Ma'], errors='coerce').median() if 'Age_Ma' in g else np.nan,
                'Target_Type':'arc segment',
                arc_col:name,
            })
        return pd.DataFrame(rows)
    if target_mode == 'CRUST1.0 grid cell':
        grid = read_default_crust_grid(str(DEFAULT_CRUST1_GRID))
        if grid.empty or not {'Lat','Lon','CRUST1_Total_Crust_km'}.issubset(grid) or not {'Lat','Lon'}.issubset(samples):
            return pd.DataFrame()
        pts = samples.dropna(subset=['Lat','Lon']).copy()
        glat = grid['Lat'].to_numpy(dtype=float)
        glon = grid['Lon'].to_numpy(dtype=float)
        # Chunked broadcast haversine: replaces per-row iterrows + scalar
        # haversine_km calls. Chunk size bounds the temporary N_chunk × M
        # distance matrix to ~M*500*8 ≈ 260 MB for an M ≈ 64.8k CRUST1 grid,
        # avoiding the multi-GB peak a full-N broadcast would need.
        plat = pd.to_numeric(pts['Lat'], errors='coerce').to_numpy(dtype=float)
        plon = pd.to_numeric(pts['Lon'], errors='coerce').to_numpy(dtype=float)
        nearest = []
        _chunk = 500
        for _start in range(0, len(plat), _chunk):
            _end = min(_start + _chunk, len(plat))
            _d = haversine_km(plat[_start:_end, None], plon[_start:_end, None],
                              glat[None, :], glon[None, :])
            nearest.extend(np.nanargmin(_d, axis=1).astype(int).tolist())
        targets = grid.iloc[sorted(set(nearest))].copy().reset_index(drop=True)
        targets['Target_ID'] = ['CRUST1 cell '+str(i+1) for i in range(len(targets))]
        targets['Age_Ma'] = 0.0
        targets['Target_Type'] = 'CRUST1.0 grid cell'
        return targets
    targets = samples.copy().reset_index(drop=True)
    targets['Target_ID'] = targets.get('Sample_ID', pd.Series(np.arange(len(targets))+1)).astype(str)
    targets['Target_Type'] = 'sample point'
    return targets

@st.cache_data(show_spinner=False)
def attach_target_domains_from_samples(targets, samples, domain_col=None, radius_col=None):
    out = targets.copy()
    if out.empty or samples.empty:
        return out
    if not {'Lat','Lon'}.issubset(out) or not {'Lat','Lon'}.issubset(samples):
        return out
    sample_coords = samples.dropna(subset=['Lat','Lon']).copy()
    if sample_coords.empty:
        return out
    sample_coords['Lat'] = pd.to_numeric(sample_coords['Lat'], errors='coerce')
    sample_coords['Lon'] = pd.to_numeric(sample_coords['Lon'], errors='coerce')
    sample_coords = sample_coords.dropna(subset=['Lat','Lon'])
    if sample_coords.empty:
        return out
    for idx, target in out.iterrows():
        tlat = pd.to_numeric(pd.Series([target.get('Lat')]), errors='coerce').iloc[0]
        tlon = pd.to_numeric(pd.Series([target.get('Lon')]), errors='coerce').iloc[0]
        if pd.isna(tlat) or pd.isna(tlon):
            continue
        nearest_idx = None
        if domain_col and domain_col in sample_coords:
            if domain_col not in out:
                out[domain_col] = pd.Series([pd.NA] * len(out), index=out.index, dtype='object')
            else:
                out[domain_col] = out[domain_col].astype('object')
            # Nearest-sample assignment. Previous "polygon from raw scatter points"
            # path was unreliable (MplPath result depended on row order) — removed.
            # Real polygon containment is handled upstream by attach_uploaded_geological_domains.
            d = haversine_km(float(tlat), float(tlon), sample_coords['Lat'].to_numpy(dtype=float), sample_coords['Lon'].to_numpy(dtype=float))
            nearest_idx = sample_coords.index[int(np.nanargmin(d))]
            out.at[idx, domain_col] = sample_coords.at[nearest_idx, domain_col]
        if radius_col and radius_col in sample_coords:
            if radius_col not in out:
                out[radius_col] = np.nan
            if domain_col and domain_col in out and pd.notna(out.at[idx, domain_col]) and domain_col in sample_coords:
                rvals = pd.to_numeric(sample_coords.loc[sample_coords[domain_col].astype(str).eq(str(out.at[idx, domain_col])), radius_col], errors='coerce').dropna()
                if not rvals.empty:
                    out.at[idx, radius_col] = float(rvals.median())
                    continue
            if nearest_idx is None:
                d = haversine_km(float(tlat), float(tlon), sample_coords['Lat'].to_numpy(dtype=float), sample_coords['Lon'].to_numpy(dtype=float))
                nearest_idx = sample_coords.index[int(np.nanargmin(d))]
            out.at[idx, radius_col] = sample_coords.at[nearest_idx, radius_col]
    return out

@st.cache_data(show_spinner=False)
def local_crustal_thickness_estimates(targets, samples, value_col='Predicted_km',
                                      time_mode='Modern benchmark: Age_Ma <= 5', target_age=0.0,
                                      time_window=10.0, age_bin_width=10.0,
                                      spatial_mode='Radius with minimum-N fallback',
                                      initial_radius_km=100.0, max_radius_km=250.0, radius_step_km=50.0,
                                      nearest_n=30, minimum_n=10, preferred_n=30,
                                      rock_types=None, domain_col=None, radius_col=None,
                                      n_boot=500, seed=42, high_iqr_km=15.0,
                                      high_bootstrap_km=10.0, high_sediment_km=5.0):
    if targets.empty or samples.empty or value_col not in samples or not {'Lat','Lon'}.issubset(targets) or not {'Lat','Lon'}.issubset(samples):
        return pd.DataFrame()
    work = samples.copy().reset_index(drop=True)
    for c in ['Lat','Lon','Age_Ma',value_col,'LOI']:
        if c in work:
            work[c] = pd.to_numeric(work[c], errors='coerce')
    work = work.dropna(subset=['Lat','Lon',value_col])
    if rock_types and 'Rock_Type_Model' in work:
        work = work[work['Rock_Type_Model'].astype(str).isin([str(x) for x in rock_types])]
    if work.empty:
        return pd.DataFrame()
    if 'Model' in work and 'Model' not in targets and work['Model'].nunique(dropna=True) > 1:
        frames = []
        for model_name, model_work in work.groupby('Model', dropna=False):
            model_targets = targets.copy()
            model_targets['Model'] = model_name
            frames.append(local_crustal_thickness_estimates(
                model_targets, model_work, value_col, time_mode, target_age, time_window, age_bin_width,
                spatial_mode, initial_radius_km, max_radius_km, radius_step_km, nearest_n, minimum_n,
                preferred_n, rock_types, domain_col, radius_col, n_boot, seed, high_iqr_km,
                high_bootstrap_km, high_sediment_km
            ))
        return tidy_numbers(pd.concat([f for f in frames if not f.empty], ignore_index=True)) if frames else pd.DataFrame()
    rows = []
    for i, target in targets.reset_index(drop=True).iterrows():
        tlat = pd.to_numeric(pd.Series([target.get('Lat')]), errors='coerce').iloc[0]
        tlon = pd.to_numeric(pd.Series([target.get('Lon')]), errors='coerce').iloc[0]
        if pd.isna(tlat) or pd.isna(tlon):
            continue
        pool = work.copy()
        model_name = target.get('Model', 'all')
        if 'Model' in pool and pd.notna(model_name) and str(model_name) != 'all':
            pool = pool[pool['Model'].astype(str).eq(str(model_name))]
        target_age_value = pd.to_numeric(pd.Series([target.get('Age_Ma', target_age)]), errors='coerce').iloc[0]
        used_time_window = np.nan
        if time_mode == 'Modern benchmark: Age_Ma <= 5' and 'Age_Ma' in pool:
            pool = pool[pool['Age_Ma'] <= 5]
            used_time_window = 5.0
        elif time_mode == 'Fixed time slice' and 'Age_Ma' in pool and pd.notna(target_age_value):
            used_time_window = float(time_window)
            pool = pool[pool['Age_Ma'].sub(float(target_age_value)).abs() <= used_time_window / 2]
        elif time_mode == 'Age bins' and 'Age_Ma' in pool and pd.notna(target_age_value):
            width = max(float(age_bin_width), 1.0)
            lo = np.floor(float(target_age_value) / width) * width
            hi = lo + width
            used_time_window = width
            pool = pool[(pool['Age_Ma'] >= lo) & (pool['Age_Ma'] < hi)]
        elif time_mode == 'Ignore age':
            used_time_window = np.nan
        if domain_col and domain_col in pool.columns and domain_col in target.index and pd.notna(target.get(domain_col)):
            pool = pool[pool[domain_col].astype(str).eq(str(target.get(domain_col)))]
        if pool.empty:
            selected = pool.copy()
            used_radius = np.nan
            expanded_radius = False
            fallback_used = 'no estimate'
        else:
            pool = pool.copy()
            pool['Local_Distance_km'] = haversine_km(float(tlat), float(tlon), pool['Lat'].to_numpy(dtype=float), pool['Lon'].to_numpy(dtype=float))
            local_max_radius = float(max_radius_km)
            if radius_col and radius_col in target.index and pd.notna(target.get(radius_col)):
                local_max_radius = max(float(initial_radius_km), float(target.get(radius_col)))
            selected = pd.DataFrame()
            used_radius = np.nan
            expanded_radius = False
            fallback_used = ''
            if spatial_mode == 'Nearest N':
                selected = pool.sort_values('Local_Distance_km').head(int(nearest_n)).copy()
                used_radius = float(selected['Local_Distance_km'].max()) if not selected.empty else np.nan
                fallback_used = 'nearest N'
            elif spatial_mode == 'Radius window':
                selected = pool[pool['Local_Distance_km'] <= float(initial_radius_km)].copy()
                used_radius = float(initial_radius_km)
                fallback_used = 'radius window'
            else:
                radius = float(initial_radius_km)
                while radius <= local_max_radius + 1e-9:
                    selected = pool[pool['Local_Distance_km'] <= radius].copy()
                    used_radius = radius
                    if len(selected) >= int(minimum_n):
                        break
                    radius += float(radius_step_km)
                expanded_radius = used_radius > float(initial_radius_km)
                fallback_used = 'radius expanded' if expanded_radius else 'initial radius'
                if len(selected) < int(minimum_n) and int(preferred_n) > len(selected):
                    selected = pool.sort_values('Local_Distance_km').head(int(preferred_n)).copy()
                    used_radius = float(selected['Local_Distance_km'].max()) if not selected.empty else used_radius
                    fallback_used = 'nearest preferred N'
        stats = local_h_stats(selected[value_col] if value_col in selected else [], n_boot=n_boot, seed=seed+i)
        row = {c:target.get(c) for c in targets.columns}
        row['Model'] = model_name
        local_context_cols = ['Predicted_km','Observed_km','Residual_km','Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2'] + [p for p in PROXY_THICKNESS_LABELS if p in selected]
        for context_col in dict.fromkeys([c for c in local_context_cols if c in selected]):
            vals = pd.to_numeric(selected[context_col], errors='coerce').dropna()
            if not vals.empty:
                row[f'Local_Median_{context_col}'] = float(vals.median())
        row.update(stats)
        row.update({
            'Target_Type':target.get('Target_Type','target'),
            'Target_ID':target.get('Target_ID',i+1),
            'Estimate_Value_Column':value_col,
            'Time_Mode':time_mode,
            'Spatial_Mode':spatial_mode,
            'Local_Radius_km':used_radius,
            'Local_Time_Window_Ma':used_time_window,
            'Local_Fallback':fallback_used,
            'Local_Domain_Column':domain_col or 'none',
            'Local_Domain_Value':target.get(domain_col) if domain_col and domain_col in target.index else 'all',
            'Local_Radius_Source':radius_col or 'manual',
            'Local_Domain_Long_Axis_km':target.get(radius_col) if radius_col and radius_col in target.index else np.nan,
            'Low_N':stats['Local_N'] < int(minimum_n),
            'Expanded_Radius':bool(expanded_radius),
            'High_IQR':pd.notna(stats['Local_IQR_H']) and stats['Local_IQR_H'] > float(high_iqr_km),
            'High_Bootstrap_Uncertainty':pd.notna(stats['Bootstrap_95CI_Low_Median']) and pd.notna(stats['Bootstrap_95CI_High_Median']) and (stats['Bootstrap_95CI_High_Median'] - stats['Bootstrap_95CI_Low_Median']) > float(high_bootstrap_km),
            'No_Estimate':stats['Local_N'] == 0,
        })
        if 'CRUST1_Total_Crust_km' in row and pd.notna(row.get('CRUST1_Total_Crust_km')):
            row['Residual_Local_vs_CRUST1_Total_km'] = row['Local_Median_H_km'] - row['CRUST1_Total_Crust_km']
        if 'CRUST1_Crystalline_Crust_km' in row and pd.notna(row.get('CRUST1_Crystalline_Crust_km')):
            row['Residual_Local_vs_CRUST1_Crystalline_km'] = row['Local_Median_H_km'] - row['CRUST1_Crystalline_Crust_km']
        if 'Observed_km' in row and pd.notna(row.get('Observed_km')):
            row['Residual_Local_vs_Known_Thickness_km'] = row['Local_Median_H_km'] - row['Observed_km']
        if 'CRUST1_Total_Crust_km' in row and 'CRUST1_Crystalline_Crust_km' in row and pd.notna(row.get('CRUST1_Total_Crust_km')) and pd.notna(row.get('CRUST1_Crystalline_Crust_km')):
            row['CRUST1_Sediment_km'] = row['CRUST1_Total_Crust_km'] - row['CRUST1_Crystalline_Crust_km']
        row['High_Sediment_Cover'] = pd.notna(row.get('CRUST1_Sediment_km', np.nan)) and row.get('CRUST1_Sediment_km', 0) > float(high_sediment_km)
        row['Good_Local_Estimate'] = not (row['Low_N'] or row['High_IQR'] or row['High_Bootstrap_Uncertainty'] or row['No_Estimate'])
        rows.append(row)
    return tidy_numbers(pd.DataFrame(rows))

def crust_summaries(matched):
    group = 'Arc_or_Segment' if 'Arc_or_Segment' in matched else 'Dataset' if 'Dataset' in matched else None
    residual_cols = [c for c in ['Residual_Guo_vs_CRUST1_km','Residual_Sundell_vs_CRUST1_km','Residual_Zou_vs_CRUST1_km'] if c in matched]
    rows = []
    for c in residual_cols:
        s = pd.to_numeric(matched[c], errors='coerce').dropna()
        rows.append({'Residual':c,'n':len(s),'median_km':s.median() if len(s) else np.nan,'mean_km':s.mean() if len(s) else np.nan,'mae_km':s.abs().mean() if len(s) else np.nan,'rmse_km':np.sqrt(np.mean(s**2)) if len(s) else np.nan})
    residual_summary = pd.DataFrame(rows)
    if group:
        arc_summary = matched.groupby(group, dropna=False)[residual_cols].agg(['count','median','mean']).reset_index()
        arc_summary.columns = ['_'.join([str(x) for x in col if x]) for col in arc_summary.columns]
    else:
        arc_summary = pd.DataFrame()
    return arc_summary, residual_summary

def geojson_numeric_properties(geojson_obj):
    rows = []
    for feat in geojson_obj.get('features', []):
        rows.append(feat.get('properties', {}) or {})
    props = pd.DataFrame(rows)
    return [c for c in props.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(props[c], errors='coerce'))]

def add_crust_background_to_geofig(fig, uploaded, key_prefix='crust_bg'):
    if uploaded is None:
        return fig
    name = getattr(uploaded,'name','').lower()
    if name.endswith('.lyrx'):
        st.warning('ArcGIS .lyrx files store layer references/styles. Export the CRUST1.0 layer to GeoJSON, or upload a prepared CRUST1.0 CSV/XLSX grid with Lat, Lon, and thickness.')
        return fig
    try:
        if name.endswith(('.geojson','.json')):
            geojson_obj = json.loads(uploaded.getvalue().decode('utf-8'))
            numeric_props = geojson_numeric_properties(geojson_obj)
            if not numeric_props:
                st.warning('The GeoJSON layer loaded, but no numeric property was found to colour by.')
                return fig
            value_col = st.selectbox('CRUST1.0 background value',numeric_props,index=0,key=f'{key_prefix}_geojson_value')
            values = []
            locations = []
            for i, feat in enumerate(geojson_obj.get('features', [])):
                feat['id'] = str(i)
                locations.append(str(i))
                values.append(pd.to_numeric(feat.get('properties', {}).get(value_col), errors='coerce'))
            fig.add_trace(go.Choropleth(
                geojson=geojson_obj,locations=locations,z=values,name='CRUST1.0 background',
                colorscale='Viridis',marker_line_width=0,opacity=0.32,colorbar_title=value_col,
                hovertemplate=f'{value_col}=%{{z:.1f}}<extra>CRUST1.0</extra>'
            ))
            fig.data = (fig.data[-1],) + fig.data[:-1]
            return fig
        grid = read_crust_grid(uploaded)
        bg_cols = [c for c in ['CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Vp','CRUST1_Density'] if c in grid]
        if not {'Lat','Lon'}.issubset(grid) or not bg_cols:
            st.warning('CRUST1.0 background grid needs Lat, Lon, and at least one thickness/velocity/density column.')
            return fig
        value_col = st.selectbox('CRUST1.0 background value',bg_cols,index=0,key=f'{key_prefix}_grid_value')
        bg = grid.dropna(subset=['Lat','Lon',value_col]).copy()
        fig.add_trace(go.Scattergeo(
            lat=bg['Lat'],lon=bg['Lon'],mode='markers',name='CRUST1.0 background',
            marker=dict(size=4,color=bg[value_col],colorscale='Viridis',opacity=0.28,colorbar=dict(title=value_col)),
            hovertemplate=f'Lat=%{{lat:.2f}}<br>Lon=%{{lon:.2f}}<br>{value_col}=%{{marker.color:.1f}}<extra>CRUST1.0</extra>'
        ))
        fig.data = (fig.data[-1],) + fig.data[:-1]
    except Exception as e:
        st.warning(f'Could not load CRUST1.0 background layer: {e}')
    return fig

@st.cache_data(show_spinner=False)
def resample_to_display_grid(grid_df, value_col, grid_id):
    """Resample any reference grid to a regular 1°×1° surface for uniform map display."""
    lats = np.arange(-89.5, 90.0, 1.0)
    lons = np.arange(-179.5, 180.0, 1.0)
    lg, lo = np.meshgrid(lats, lons, indexing='ij')
    flat_lats = lg.ravel(); flat_lons = lo.ravel()
    vals = grid_interpolate(flat_lats, flat_lons, grid_df, value_col, grid_id)
    ok = np.isfinite(vals)
    return pd.DataFrame({'Lat': flat_lats[ok], 'Lon': flat_lons[ok], value_col: vals[ok]})

def add_grid_background_to_geofig(fig, grid, value_col, label, grid_id='grid', display_mode='Surface', colorbar_x=-0.08):
    """Add a reference grid as a background layer. display_mode: 'Surface' (resampled 1°, square tiles) or 'Points' (raw nodes)."""
    if grid is None or grid.empty or not {'Lat','Lon',value_col}.issubset(grid):
        return fig
    if display_mode == 'Surface':
        # Resample to uniform 1°×1° grid. Use square markers sized ~14px so adjacent tiles
        # overlap at any normal zoom level — squares tile gap-free unlike circles.
        bg = resample_to_display_grid(grid, value_col, grid_id)
        marker_size, marker_symbol, opacity = 14, 'square', 0.50
    else:
        bg = grid.dropna(subset=['Lat','Lon',value_col]).copy()
        n_pts = len(bg)
        marker_size = 4 if n_pts > 40000 else (6 if n_pts > 10000 else 9)
        marker_symbol, opacity = 'circle', 0.60
    short_label = label.replace(' / Alfonso 2019','').replace(' (Alfonso 2019)','')
    fig.add_trace(go.Scattergeo(
        lat=bg['Lat'], lon=bg['Lon'], mode='markers', name=short_label,
        marker=dict(
            size=marker_size, symbol=marker_symbol,
            color=bg[value_col], colorscale=THICKNESS_COLORSCALE,
            cmin=THICKNESS_CMIN, cmax=THICKNESS_CMAX, opacity=opacity,
            colorbar=dict(title=f'{label}:<br>crustal thickness<br>[Km]', x=colorbar_x, len=0.72, y=0.50),
        ),
        hovertemplate=f'Lat=%{{lat:.2f}}<br>Lon=%{{lon:.2f}}<br>{short_label}=%{{marker.color:.1f}} Km<extra></extra>',
    ))
    fig.data = (fig.data[-1],) + fig.data[:-1]
    return fig

def add_crust_grid_background_to_geofig(fig, grid, value_col='CRUST1_Total_Crust_km', colorbar_x=-0.08):
    return add_grid_background_to_geofig(fig, grid, value_col, 'CRUST1.0', 'CRUST1.0', 'Surface', colorbar_x)

def model_map_legend_title(color_by):
    if color_by == 'Predicted_km':
        return 'model:<br>crustal thickness<br>[Km]'
    if color_by == 'Observed_km':
        return 'known:<br>crustal thickness<br>[Km]'
    if color_by == 'Residual_km':
        return 'model:<br>residual<br>[Km]'
    if str(color_by).startswith('Modelled_') or color_by in ['H_Guo_ERT_km','Preferred_H_km']:
        return 'model:<br>crustal thickness<br>[Km]'
    if color_by == 'Crust_Thickness':
        return 'known:<br>crustal thickness<br>[Km]'
    if color_by == 'Age_Ma':
        return 'model:<br>age [Ma]'
    if color_by in ['Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label']:
        return f'model:<br>{color_by.replace("_"," ")}'
    if color_by in ['Rock_Type_Model','Rock_Type']:
        return 'model:<br>rock type'
    if color_by in ['Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset']:
        return f'model:<br>{color_by.replace("_"," ")}'
    return f'model:<br>{str(color_by).replace("_"," ")}'

def model_map_option_label(color_by, modelled_count=1):
    if str(color_by).startswith('Modelled_') and modelled_count > 1:
        label = str(color_by).replace('Modelled_','').replace('_km','').replace('_',' ')
        return f'model: crustal thickness [Km] - {label}'
    return model_map_legend_title(color_by).replace('<br>',' ')

def map_hover_data(df, cols):
    out = {}
    for c in cols:
        if c not in df:
            continue
        if pd.api.types.is_numeric_dtype(col_series(df, c)) or numeric_series(df, c).notna().any():
            out[c] = ':.1f'
        else:
            out[c] = True
    return out

def plotly_selected_indices(event):
    if not event:
        return []
    try:
        selection = event.get('selection', {}) if isinstance(event, dict) else getattr(event, 'selection', {})
        points = selection.get('points', []) if isinstance(selection, dict) else getattr(selection, 'points', [])
    except Exception:
        return []
    idx = []
    for p in points or []:
        if isinstance(p, dict):
            val = p.get('point_index', p.get('pointIndex', p.get('point_number', p.get('pointNumber'))))
        else:
            val = getattr(p, 'point_index', getattr(p, 'pointNumber', None))
        if val is not None:
            try:
                idx.append(int(val))
            except Exception:
                pass
    return sorted(set(idx))


def _to_hex_color(c, fallback='#888888'):
    """Coerce any Plotly-palette colour string to a 6-digit hex code.

    Plotly's qualitative palettes mix two formats: ``Set2`` returns
    ``rgb(102, 194, 165)`` style triples while ``D3``/``Plotly`` already
    return ``#1f77b4`` hex. ``st.color_picker`` only accepts hex, so any
    rgb(...) value coming from a Bold/Set2 palette must be converted
    first or Streamlit raises ``StreamlitAPIException``.

    Accepts: ``#abcdef``, ``#abc``, ``rgb(r, g, b)``, ``rgba(r, g, b, a)``.
    Returns the 6-digit hex form; falls back to ``fallback`` on parse
    failure so calls never raise.
    """
    if c is None:
        return fallback
    s = str(c).strip()
    if not s:
        return fallback
    if s.startswith('#'):
        # Expand short form #abc → #aabbcc; pass-through 6-digit
        if len(s) == 4:
            return '#' + ''.join(ch * 2 for ch in s[1:])
        if len(s) == 7:
            return s
        return fallback
    if s.lower().startswith(('rgb(', 'rgba(')) and ')' in s:
        try:
            inner = s[s.index('(') + 1:s.index(')')]
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            r, g, b = (int(round(float(parts[i]))) for i in range(3))
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            return f'#{r:02x}{g:02x}{b:02x}'
        except Exception:
            return fallback
    return fallback


def plotly_polygon_indices(event, df, x_col, y_col):
    """Return df.index values whose (x_col, y_col) coordinates fall inside
    *any* lasso polygon or selection box reported by a Streamlit
    plotly_chart on_select event.

    Why this exists: ``plotly_selected_indices`` only returns
    ``point_index`` for points actually drawn in the chart's trace.  When
    the chart is downsampled for performance (e.g. 10 K of 90 K points
    drawn), the user's lasso silently misses the ~80 K hidden bench rows
    that are physically inside the polygon.  This helper instead does a
    geometric test against the *full* dataframe — so every point inside
    the polygon is captured, regardless of whether it was drawn.

    Robustness:
      • Tries ``x``/``y`` first (cartesian Scatter / Scattergl), then
        ``lon``/``lat`` (Scattergeo / mapbox), then ``lng``/``lat``.
      • Boxes are 2-element ``x``/``y`` ranges → axis-aligned rectangle.
      • Lasso polygons need ≥ 3 vertices; ones with fewer are skipped.
      • If matplotlib isn't importable, returns an empty set so callers
        can safely fall back to ``plotly_selected_indices``.
    """
    if event is None or df is None or len(df) == 0:
        return set()
    if x_col not in df.columns or y_col not in df.columns:
        return set()
    sel = (event.get('selection', {}) if isinstance(event, dict)
           else getattr(event, 'selection', None))
    if not sel:
        return set()
    if isinstance(sel, dict):
        lassos = sel.get('lasso', []) or []
        boxes  = sel.get('box',   []) or []
    else:
        lassos = getattr(sel, 'lasso', []) or []
        boxes  = getattr(sel, 'box',   []) or []
    if not lassos and not boxes:
        return set()

    def _xy(d):
        """Pull (x_arr, y_arr) from a polygon/box dict, trying all the
        coordinate-key conventions Plotly uses across trace types."""
        if not isinstance(d, dict):
            return None, None
        for xk, yk in (('x', 'y'), ('lon', 'lat'),
                       ('lng', 'lat'), ('longitude', 'latitude')):
            if xk in d and yk in d:
                return d[xk], d[yk]
        return None, None

    xs = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
    ys = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not valid.any():
        return set()
    pts = np.column_stack([xs, ys])
    idx_arr = df.index.to_numpy()
    inside = np.zeros(len(df), dtype=bool)

    try:
        from matplotlib.path import Path as _MPath
    except Exception:
        return set()

    for poly in lassos:
        px_arr, py_arr = _xy(poly)
        if px_arr is None or py_arr is None:
            continue
        try:
            px_arr = np.asarray(px_arr, dtype=float)
            py_arr = np.asarray(py_arr, dtype=float)
        except Exception:
            continue
        if len(px_arr) < 3 or len(py_arr) < 3:
            continue
        try:
            path = _MPath(np.column_stack([px_arr, py_arr]))
            mask = np.zeros(len(df), dtype=bool)
            mask[valid] = path.contains_points(pts[valid])
            inside |= mask
        except Exception:
            continue

    for box in boxes:
        bx, by = _xy(box)
        if bx is None or by is None:
            continue
        try:
            bx = [float(v) for v in bx]
            by = [float(v) for v in by]
        except Exception:
            continue
        if len(bx) < 2 or len(by) < 2:
            continue
        x0, x1 = sorted([bx[0], bx[1]])
        y0, y1 = sorted([by[0], by[1]])
        mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1) & valid
        inside |= mask

    return set(int(i) for i in idx_arr[inside])

def style_training_map_legends(fig, color_by):
    title = model_map_legend_title(color_by)
    fig.update_layout(
        coloraxis_colorbar=dict(title=title,x=1.02,len=0.72,y=0.50),
        legend=dict(title=dict(text=title.replace('<br>',' ')),x=1.02,y=0.98,yanchor='top',bgcolor='rgba(255,255,255,0.78)')
    )
    return fig

def is_thickness_layer(col):
    """True when col is a km-scale crustal-thickness column that should use
    the shared THICKNESS_COLORSCALE / THICKNESS_CMIN‥CMAX range."""
    s = str(col).lower()
    return (s.endswith('_km')
            or ('crust' in s and 'thick' in s)
            or s in {'predicted_km', 'observed_km', 'h_guo_ert_km'})

def is_residual_layer(col):
    """True when col is a signed residual / delta that should use
    RESIDUAL_COLORSCALE centred on zero."""
    return str(col) in {'Residual_km', 'Delta_km', 'GAME_delta_km'}

def thickness_scatter_kw(col, df):
    """Return extra kwargs for px.scatter_geo when col is a thickness layer."""
    if is_thickness_layer(col) and col in df and pd.api.types.is_numeric_dtype(df[col]):
        return dict(
            color_continuous_scale=THICKNESS_COLORSCALE,
            range_color=[THICKNESS_CMIN, THICKNESS_CMAX],
        )
    return {}

def map_color_kw(col, df):
    """Unified color kwargs for px.scatter_geo — residual wins over thickness."""
    if is_residual_layer(col):
        return dict(
            color_continuous_scale=RESIDUAL_COLORSCALE,
            color_continuous_midpoint=0,
        )
    return thickness_scatter_kw(col, df)

def apply_geo_zoom_to_data(fig, df, enabled):
    if not enabled or df is None or df.empty or not {'Lat','Lon'}.issubset(df):
        return fig
    coords = df[['Lat','Lon']].apply(pd.to_numeric, errors='coerce').dropna()
    if coords.empty:
        return fig
    lat_min, lat_max = coords['Lat'].min(), coords['Lat'].max()
    lon_min, lon_max = coords['Lon'].min(), coords['Lon'].max()
    lat_span = max(float(lat_max - lat_min), 2.0)
    lon_span = max(float(lon_max - lon_min), 2.0)
    span = max(lat_span * 1.8, lon_span)
    scale = max(1.0, min(18.0, 170.0 / span))
    fig.update_geos(
        center=dict(lat=float(coords['Lat'].mean()), lon=float(coords['Lon'].mean())),
        projection_scale=scale
    )
    return fig

_GEO_POINT_LIMIT = 8_000

def geo_downsample(df: pd.DataFrame, limit: int = _GEO_POINT_LIMIT) -> pd.DataFrame:
    """Stratified downsample for scatter_geo; preserves spatial spread."""
    if len(df) <= limit:
        return df
    # Keep every Nth row — simple but preserves order/geography. iloc[::step]
    # and head() already return new DataFrames, so no defensive copy needed.
    step = max(1, len(df) // limit)
    return df.iloc[::step].head(limit)

def xlsx_bytes(df):
    bio=BytesIO()
    with pd.ExcelWriter(bio,engine='xlsxwriter') as w: df.to_excel(w,index=False,sheet_name='Results')
    return bio.getvalue()

def barrick_logo_html():
    logo_names = ['barrick_logo.svg','barrick_logo.png','barrick_logo.jpg','barrick_logo.jpeg']
    mime = {'svg':'image/svg+xml','png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg'}
    for name in logo_names:
        path = Path(__file__).resolve().parent / name
        if path.exists():
            ext = path.suffix.lower().lstrip('.')
            data = base64.b64encode(path.read_bytes()).decode('ascii')
            return f'<img class="app-logo-image" src="data:{mime.get(ext,"image/png")};base64,{data}" alt="Barrick logo" />'
    return '<div class="app-logo-fallback">BARRICK</div>'

st.set_page_config(page_title='Mohometer',layout='wide')
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 13px;
}
.block-container {
    padding-top: 3.0rem;
    padding-bottom: 2.0rem;
    max-width: 1180px;
}
h1 {
    font-size: 1.85rem !important;
    line-height: 1.15 !important;
    margin-bottom: 1.2rem !important;
}
h2 {
    font-size: 1.35rem !important;
    line-height: 1.2 !important;
    margin-top: 1.0rem !important;
}
h3 {
    font-size: 1.05rem !important;
}
p, label, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] {
    font-size: 0.88rem !important;
}
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.55rem 0.65rem;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    color: #64748b !important;
    margin-bottom: 0.15rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    line-height: 1.15 !important;
}
[data-testid="stMetricValue"] div {
    font-size: inherit !important;
    line-height: inherit !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
[data-testid="stSelectbox"] div,
[data-testid="stMultiSelect"] div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    font-size: 0.86rem !important;
}
[data-testid="stExpander"] {
    border-radius: 6px !important;
}
[data-testid="stExpander"] details summary p {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}
button, [role="tab"] {
    font-size: 0.82rem !important;
}
[data-testid="stDataFrame"] {
    font-size: 0.78rem !important;
}
.stPlotlyChart {
    font-size: 0.82rem;
}
.summary-tile {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.58rem 0.68rem;
    min-height: 3.25rem;
}
.summary-tile.compact { display: inline-block; min-width: 7rem; min-height: 2.45rem; padding: 0.42rem 0.58rem; }
.summary-tile .summary-label {
    color: #64748b;
    font-size: 0.72rem;
    line-height: 1.05;
    margin-bottom: 0.35rem;
}
.summary-tile .summary-value {
    color: #1f2937;
    font-size: 0.98rem;
    line-height: 1.22;
    overflow-wrap: anywhere;
}
.summary-tile .summary-number {
    font-size: 1.02rem;
    font-variant-numeric: tabular-nums;
}
.app-masthead {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 1.25rem;
    align-items: start;
    border-bottom: 1px solid #e5e7eb;
    padding-top: 0.45rem;
    padding-bottom: 0.85rem;
    margin-bottom: 0.85rem;
}
.app-title {
    font-size: 2.0rem;
    line-height: 1.0;
    font-weight: 700;
    color: #111827;
    margin: 0 0 0.28rem 0;
    letter-spacing: 0;
}
.app-subtitle {
    font-size: 0.98rem;
    color: #374151;
    margin: 0.08rem 0;
}
.app-author {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.45rem;
}
.app-methods {
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 0.34rem;
    max-width: 58rem;
}
.app-logo-box {
    text-align: right;
    min-width: 12rem;
}
.app-logo-image {
    max-width: 11.5rem;
    max-height: 3.4rem;
    object-fit: contain;
}
.app-logo-fallback {
    display: inline-block;
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    color: #111827;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 0.38rem 0.6rem;
}
.app-program {
    color: #9ca3af;
    font-size: 0.68rem;
    margin-top: 0.24rem;
    white-space: nowrap;
}
.app-credit {
    color: #6b7280;
    font-size: 0.70rem;
    line-height: 1.25;
    margin-top: 0.28rem;
}

/* ── Miner loading overlay ── */
@keyframes miner-fadeout {
  0%   { opacity:1; pointer-events:all; }
  80%  { opacity:1; }
  100% { opacity:0; pointer-events:none; }
}
@keyframes pickaxe-swing {
  0%,100% { transform: rotate(-35deg); }
  50%      { transform: rotate(25deg);  }
}
@keyframes spark-fly {
  0%   { opacity:1; transform: translate(0,0) scale(1); }
  100% { opacity:0; transform: translate(14px,-8px) scale(0.3); }
}
@keyframes dust-rise {
  0%   { opacity:0.7; transform: translate(0,0) scale(1); }
  100% { opacity:0;   transform: translate(-8px,-12px) scale(1.6); }
}
@keyframes progress-bar {
  0%   { width:0%; }
  100% { width:92%; }
}
#miner-loader {
  position:fixed; inset:0; z-index:999999;
  background:#ffffff;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:18px;
  animation: miner-fadeout 3.4s ease-in-out forwards;
}
#miner-loader .loader-title {
  font-family:'Segoe UI',Arial,sans-serif;
  font-size:1.55rem; font-weight:700;
  color:#111827; letter-spacing:0.02em;
}
#miner-loader .loader-sub {
  font-family:'Segoe UI',Arial,sans-serif;
  font-size:0.82rem; color:#6b7280;
  margin-top:-12px;
}
#miner-loader .loader-bar-wrap {
  width:220px; height:5px; background:#f3f4f6;
  border-radius:99px; overflow:hidden;
}
#miner-loader .loader-bar {
  height:100%; background:linear-gradient(90deg,#F59E0B,#D97706);
  border-radius:99px;
  animation: progress-bar 3.0s cubic-bezier(.4,0,.2,1) forwards;
}
.arm-group {
  transform-origin: 52px 60px;
  animation: pickaxe-swing 0.55s ease-in-out infinite alternate;
}
.spark1 { animation: spark-fly 0.55s ease-out infinite 0.0s; }
.spark2 { animation: spark-fly 0.55s ease-out infinite 0.18s; }
.spark3 { animation: spark-fly 0.55s ease-out infinite 0.30s; }
.dust1  { animation: dust-rise 0.7s ease-out infinite 0.05s; }
.dust2  { animation: dust-rise 0.7s ease-out infinite 0.22s; }

/* ── Running-state overlay (shown by JS when Streamlit is computing) ── */
#miner-running {
  position:fixed; inset:0; z-index:999998;
  background:rgba(255,255,255,0.82);
  backdrop-filter:blur(2px);
  display:none; flex-direction:column;
  align-items:center; justify-content:center; gap:14px;
  pointer-events:none;
  transition: opacity 0.18s ease;
}
#miner-running.active { display:flex; }
#miner-running .r-title {
  font-family:'Segoe UI',Arial,sans-serif;
  font-size:1.05rem; font-weight:600;
  color:#374151; letter-spacing:0.01em;
}
#miner-running .r-sub {
  font-family:'Segoe UI',Arial,sans-serif;
  font-size:0.75rem; color:#9ca3af;
  margin-top:-10px;
}
</style>
""", unsafe_allow_html=True)

# Shared SVG miner — used in both the initial loader and the running overlay
_MINER_SVG = """
  <svg width="110" height="110" viewBox="0 0 130 130" xmlns="http://www.w3.org/2000/svg">
    <rect x="15" y="108" width="100" height="6" rx="3" fill="#D97706" opacity="0.25"/>
    <rect x="88" y="72" width="30" height="36" rx="4" fill="#9CA3AF" opacity="0.55"/>
    <rect x="88" y="72" width="30" height="36" rx="4" fill="none" stroke="#6B7280" stroke-width="1.2"/>
    <path d="M 98 80 l 4 6 l -3 4 l 5 8" stroke="#374151" stroke-width="1" fill="none" opacity="0.4"/>
    <circle class="spark1" cx="90" cy="96" r="2.2" fill="#FCD34D"/>
    <circle class="spark2" cx="92" cy="94" r="1.6" fill="#F59E0B"/>
    <circle class="spark3" cx="89" cy="97" r="1.4" fill="#FEF3C7"/>
    <circle class="dust1" cx="88" cy="100" r="4" fill="#D1D5DB" opacity="0.5"/>
    <circle class="dust2" cx="84" cy="98"  r="3" fill="#E5E7EB" opacity="0.4"/>
    <line x1="52" y1="90" x2="40" y2="108" stroke="#374151" stroke-width="3" stroke-linecap="round"/>
    <line x1="52" y1="90" x2="64" y2="108" stroke="#374151" stroke-width="3" stroke-linecap="round"/>
    <line x1="52" y1="60" x2="52" y2="90" stroke="#374151" stroke-width="3" stroke-linecap="round"/>
    <line x1="52" y1="68" x2="38" y2="82" stroke="#374151" stroke-width="2.5" stroke-linecap="round"/>
    <g class="arm-group">
      <line x1="52" y1="60" x2="76" y2="74" stroke="#374151" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="76" y1="74" x2="95" y2="92" stroke="#92400E" stroke-width="2" stroke-linecap="round"/>
      <path d="M 90 88 L 104 80 L 100 96 Z" fill="#4B5563" stroke="#374151" stroke-width="0.8"/>
      <path d="M 90 88 L 80 80 L 84 92 Z" fill="#6B7280" stroke="#374151" stroke-width="0.8"/>
    </g>
    <circle cx="52" cy="48" r="11" fill="#374151"/>
    <ellipse cx="52" cy="42" rx="16" ry="4" fill="#F59E0B"/>
    <path d="M 38 42 Q 52 24 66 42 Z" fill="#FBBF24" stroke="#D97706" stroke-width="0.8"/>
    <circle cx="52" cy="33" r="3" fill="#FEF3C7" stroke="#D97706" stroke-width="0.8"/>
    <path d="M 52 30 l -6 -5 M 52 30 l 0 -6 M 52 30 l 6 -5" stroke="#FCD34D" stroke-width="1" stroke-linecap="round" opacity="0.7"/>
  </svg>"""

st.markdown(f"""
<div id="miner-loader">
  {_MINER_SVG}
  <div class="loader-title">⛏&thinsp; Mohometer</div>
  <div class="loader-sub">Loading crustal architecture tools&hellip;</div>
  <div class="loader-bar-wrap"><div class="loader-bar"></div></div>
</div>

<div id="miner-running">
  {_MINER_SVG}
  <div class="r-title">⛏&thinsp; Working on it&hellip;</div>
  <div class="r-sub">Crunching the crust</div>
</div>

<script>
(function() {{
  var overlay = document.getElementById('miner-running');
  if (!overlay) return;

  function isRunning() {{
    // 1. data-stale="true" — what causes the grey-out on widgets during rerun
    if (document.querySelector('[data-stale="true"]')) return true;

    // 2. Streamlit status widget (top-right "Running…" indicator)
    var w = document.querySelector('[data-testid="stStatusWidget"]');
    if (w) {{
      var txt = (w.getAttribute('aria-label') || w.innerText || w.textContent || '').toLowerCase();
      if (txt.includes('running') || txt.includes('processing') || txt.includes('rerun')) return true;
      // class-based fallback (newer Streamlit uses className like StatusWidget--running)
      var cls = (w.className && (w.className.baseVal || w.className.toString()) || '').toLowerCase();
      if (cls.includes('running')) return true;
      // Children may carry the running class
      if (w.querySelector('.running, [class*="running" i], [class*="Running"]')) return true;
    }}

    // 3. Inline spinner / blocking spinner
    if (document.querySelector('[data-testid="stSpinner"]')) return true;
    if (document.querySelector('[data-testid="stConnectionStatus"][data-state="running"]')) return true;

    // 4. App-level running attribute (some Streamlit builds set this on stApp)
    var app = document.querySelector('[data-testid="stApp"]');
    if (app) {{
      var st = (app.getAttribute('data-test-script-state') || app.getAttribute('data-script-state') || '').toLowerCase();
      if (st === 'running' || st === 'rerun_requested') return true;
    }}

    return false;
  }}

  var debounce;
  function update() {{
    clearTimeout(debounce);
    debounce = setTimeout(function() {{
      if (isRunning()) {{
        overlay.classList.add('active');
      }} else {{
        overlay.classList.remove('active');
      }}
    }}, 30);
  }}

  var observer = new MutationObserver(update);
  observer.observe(document.body, {{
    childList: true, subtree: true,
    attributes: true,
    attributeFilter: ['aria-label', 'data-testid', 'class', 'data-stale',
                      'data-test-script-state', 'data-script-state', 'data-state']
  }});

  // Periodic safety check in case mutations are missed
  setInterval(update, 350);
  update();
}})();
</script>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="app-masthead">
  <div>
    <div class="app-title">Mohometer</div>
    <div class="app-subtitle">Crustal thickness estimation</div>
    <div class="app-methods">Methods integrated from Guo &amp; Yang (2023), Zou et al. (2021), Sundell et al. (2021), Profeta et al. (2015), Mantle &amp; Collins (2008), Luffi &amp; Ducea (2022), and CRUST1.0.</div>
  </div>
  <div class="app-logo-box">
    {barrick_logo_html()}
    <div class="app-program">&lt;ASC - Regional - Crustal Architecture - Crustal Thickness&gt;</div>
    <div class="app-credit">Holder (2026)<br>david.holder@barrick.com</div>
  </div>
</div>
""", unsafe_allow_html=True)

la_mode='raw_ppm'
st.sidebar.slider('Point size (all charts)', 2, 14, 5, 1, key='global_point_size')
pred_no_header=False

# ── Workflow breadcrumb (appears once Prepare has data) ─────────────────────
def _bc_step(label, done, note=''):
    col  = '#16a34a' if done else '#9ca3af'
    tick = '✓' if done else '○'
    base = (f'<span style="color:{col};font-weight:{"600" if done else "400"};'
            f'white-space:nowrap">{tick} {label}</span>')
    if note:
        base += (f'<span style="color:#9ca3af;font-size:11px;margin-left:3px">'
                 f'{note}</span>')
    return base

_bc_prepare = bool(get_prepped_pool())
if _bc_prepare:
    _bc_group   = any('Group_ID' in df.columns for df in get_prepped_pool().values())
    _bc_model   = bool(st.session_state.get('_training_sources_used'))
    _bc_predict = ('_rs_pred_bench' in st.session_state
                   or '_rs_val_bench' in st.session_state)
    _bc_arrow   = '<span style="color:#d1d5db;margin:0 6px">→</span>'
    st.markdown(
        '<div style="display:flex;gap:0;align-items:center;padding:4px 0 12px 2px;'
        'font-size:13px;font-family:sans-serif;flex-wrap:wrap">'
        + _bc_step('Prepare', _bc_prepare)
        + _bc_arrow
        + _bc_step('Group', _bc_group, '(optional)')
        + _bc_arrow
        + _bc_step('Model', _bc_model)
        + _bc_arrow
        + _bc_step('Validate / Predict', _bc_predict)
        + '</div>',
        unsafe_allow_html=True,
    )
seed=42

def summary_tile(label, value, numeric=False, compact=False):
    value_class = 'summary-value summary-number' if numeric else 'summary-value'
    tile_class = 'summary-tile compact' if compact else 'summary-tile'
    st.markdown(f'<div class="{tile_class}"><div class="summary-label">{label}</div><div class="{value_class}">{value}</div></div>', unsafe_allow_html=True)

@st.fragment
def _render_grouping_display(prefix):
    """Fragment: map + graph display for grouping diagnostics. Reruns only on display widget changes."""
    _gmap=st.session_state.get(f'_{prefix}_group_map',pd.DataFrame())
    _ldf=st.session_state.get(f'_{prefix}_local_df',pd.DataFrame())
    _lv=st.session_state.get(f'_{prefix}_local_value','Predicted_km')
    if _gmap.empty:
        st.info('Select at least one non-manual grouping method (Preset grouping, Sample distribution long axis, or an uploaded geological domain) to show the grouping map and proxy graph.')
        return
    _gm1,_gm2,_gm3,_gm4,_gm5,_gm6=st.columns([1,1,0.5,0.5,0.5,0.5])
    _mmeths=sorted(_gmap['Grouping_Method'].dropna().astype(str).unique())
    _smeths=_gm1.multiselect('Methods',_mmeths,default=_mmeths[:1],key=f'{prefix}_disp_map_methods',help='Turn grouping methods on/off for both the map and grouping graph.')
    if len(_smeths)>1:
        st.caption('Multiple grouping methods are overlaid. Summary markers are calculated separately for each method/group, so use one method at a time when checking whether medians sit inside the sample population.')
    _fm=_gmap[_gmap['Grouping_Method'].astype(str).isin(_smeths)] if _smeths else _gmap.iloc[:0].copy()
    if not _smeths:
        st.info('Select one or more Methods to show grouping samples, medians, and graph points.')
        return
    _copts=[c for c in ['Group_ID','Grouping_Method',_lv,'Predicted_km']+[p for p in PROXY_THICKNESS_LABELS if p in _fm]+['Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset'] if c in _fm and(not pd.api.types.is_numeric_dtype(_fm[c]) or pd.to_numeric(_fm[c],errors='coerce').notna().any())]
    _copts=list(dict.fromkeys(_copts))
    _mcol=_gm2.selectbox('Colour map by',_copts,index=0,key=f'{prefix}_disp_map_color',format_func=local_option_label)
    _show_samp=_gm3.checkbox('Samples',True,key=f'{prefix}_disp_map_samples')
    _show_leg=_gm4.checkbox('Legend',False,key=f'{prefix}_disp_map_legend')
    _show_med=_gm5.checkbox('Medians',True,key=f'{prefix}_disp_map_medians')
    with st.expander('Map style',expanded=False):
        _ms1,_ms2,_ms3,_ms4=st.columns(4)
        _mpal=_ms1.selectbox('Colour palette',['Plotly','Set2','Dark24','Alphabet','Safe'],index=1,key=f'{prefix}_disp_map_palette')
        _mscale=_ms2.selectbox('Continuous scale',['Viridis','Turbo','Cividis','RdBu_r'],index=0,key=f'{prefix}_disp_map_scale')
        _ssym=_ms3.selectbox('Sample symbol',['circle','square','diamond','cross','x','triangle-up','triangle-down'],index=0,key=f'{prefix}_disp_map_ssym')
        _msym=_ms4.selectbox('Median symbol',['diamond','star','square','circle','cross','x','triangle-up'],index=0,key=f'{prefix}_disp_map_msym')
        _ms5,_ms6,_ms7=st.columns(3)
        _mcolvr=_ms5.color_picker('Median colour','#f5bf42',key=f'{prefix}_disp_map_mcol')
        _ssz=typed_slider(_ms6,'Sample size',3,14,6,1,key=f'{prefix}_disp_map_ssz')
        _sopac=typed_slider(_ms7,'Sample opacity',10,100,55,5,key=f'{prefix}_disp_map_sopac')
    _mnum=_mcol in _fm and pd.api.types.is_numeric_dtype(_fm[_mcol])
    _ghov=hover_cols(_fm,['Sample_ID','Grouping_Method','Group_ID','Age_Ma','Dataset','Arc_or_Segment','Geologic_Domain','Rock_Type_Model','Model',_lv,'Predicted_km','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb']+[p for p in PROXY_THICKNESS_LABELS if p in _fm],_mcol,'Lat','Lon')
    _fm_geo = geo_downsample(tidy_numbers(_fm))
    if len(_fm_geo) < len(_fm): st.caption(f'Map showing {len(_fm_geo):,} of {len(_fm):,} points for performance.')
    _gfig=px.scatter_geo(_fm_geo,lat='Lat',lon='Lon',color=_mcol,hover_data=map_hover_data(_fm_geo,_ghov),projection='natural earth',template='plotly_white',labels={_mcol:local_option_label(_mcol)},color_discrete_sequence=qualitative_palette(_mpal),color_continuous_scale=_mscale if _mnum else None)
    _gfig.update_traces(marker=dict(size=_ssz,symbol=_ssym,opacity=_sopac/100,line=dict(color='black',width=0.45)),selector=dict(type='scattergeo'))
    if not _show_samp:
        _gfig.for_each_trace(lambda tr:tr.update(visible=False))
    if _show_med and not _ldf.empty and {'Lat','Lon','Local_Median_H_km'}.issubset(_ldf):
        _mm=_ldf.dropna(subset=['Lat','Lon','Local_Median_H_km']).copy()
        if 'Grouping_Method' in _mm:
            _mm=_mm[_mm['Grouping_Method'].astype(str).isin(_smeths)]
        _gc2=[c for c in ['Grouping_Method','Local_Domain_Value'] if c in _mm]
        _mr=[]
        if _gc2:
            for _gk,_g in _mm.groupby(_gc2,dropna=False):
                if not isinstance(_gk,tuple): _gk=(_gk,)
                _r={c:v for c,v in zip(_gc2,_gk)}
                _r.update({'Lat':pd.to_numeric(_g['Lat'],errors='coerce').median(),'Lon':pd.to_numeric(_g['Lon'],errors='coerce').median(),'Local_Median_H_km':pd.to_numeric(_g['Local_Median_H_km'],errors='coerce').median(),'Local_N':pd.to_numeric(_g['Local_N'],errors='coerce').median() if 'Local_N' in _g else np.nan,'n_targets':len(_g)})
                _mr.append(_r)
        _mmdf=pd.DataFrame(_mr).dropna(subset=['Lat','Lon','Local_Median_H_km']) if _mr else pd.DataFrame()
        if not _mmdf.empty:
            _gnm=_mmdf[_gc2].astype(str).agg(' / '.join,axis=1) if _gc2 else pd.Series(['group median']*len(_mmdf),index=_mmdf.index)
            _gfig.add_trace(go.Scattergeo(lat=_mmdf['Lat'],lon=_mmdf['Lon'],mode='markers',name='group medians',marker=dict(symbol=_msym,size=12,color=_mcolvr,line=dict(color='black',width=1.1)),customdata=np.c_[_gnm,_mmdf['Local_Median_H_km'],_mmdf['Local_N'],_mmdf['n_targets']],hovertemplate='group=%{customdata[0]}<br>median H=%{customdata[1]:.1f} Km<br>median Local N=%{customdata[2]:.0f}<br>targets=%{customdata[3]:.0f}<extra>group median</extra>',showlegend=_show_leg))
    _gfig.update_layout(height=520,margin=dict(l=10,r=10,t=10,b=10),showlegend=_show_leg,coloraxis_showscale=_show_leg)
    st.plotly_chart(_gfig,width='stretch',key=f'{prefix}_disp_map_fig')
    st.markdown('**Grouping graph**')
    st.caption('Samples are plotted first; summary markers are calculated from those same visible rows after numeric filtering and de-duplication.')
    _gcols=[c for c in ['Predicted_km']+[p for p in PROXY_THICKNESS_LABELS if p in _fm]+['Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2'] if c in _fm and pd.to_numeric(_fm[c],errors='coerce').notna().any()]
    if len(_gcols)<2:
        st.info('Grouping graph needs at least two numeric model/proxy/chemistry columns.')
        return
    _gg1,_gg2,_gg3,_gg4,_gg5,_gg6=st.columns([1,1,1,0.65,0.55,0.75])
    _gx=_gg1.selectbox('X axis',_gcols,index=0,key=f'{prefix}_disp_graph_x',format_func=local_option_label)
    _dfy='H_Sundell2021_Paired_km' if 'H_Sundell2021_Paired_km' in _gcols and _gx!='H_Sundell2021_Paired_km' else _gcols[min(1,len(_gcols)-1)]
    _gy=_gg2.selectbox('Y axis',_gcols,index=_gcols.index(_dfy),key=f'{prefix}_disp_graph_y',format_func=local_option_label)
    _gcopts=[c for c in ['Group_ID','Grouping_Method','Predicted_km']+[p for p in PROXY_THICKNESS_LABELS if p in _fm]+['Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset'] if c in _fm and(not pd.api.types.is_numeric_dtype(_fm[c]) or pd.to_numeric(_fm[c],errors='coerce').notna().any())]
    _gcopts=list(dict.fromkeys(_gcopts))
    _gc=_gg3.selectbox('Colour graph by',_gcopts,index=0,key=f'{prefix}_disp_graph_color',format_func=local_option_label)
    _shgsamp=_gg4.checkbox('Samples',True,key=f'{prefix}_disp_graph_samples')
    _shgleg=_gg5.checkbox('Legend',False,key=f'{prefix}_disp_graph_legend')
    _gstat=_gg6.selectbox('Summary',['Median','Mean','Off'],index=0,key=f'{prefix}_disp_graph_stat')
    _gminn=typed_slider(st,'Minimum N for summary marker',1,100,10,1,key=f'{prefix}_disp_graph_minn',disabled=_gstat=='Off')
    with st.expander('Graph style',expanded=False):
        _gs1,_gs2,_gs3,_gs4=st.columns(4)
        _gpal=_gs1.selectbox('Colour palette',['Plotly','Set2','Dark24','Alphabet','Safe'],index=1,key=f'{prefix}_disp_graph_palette')
        _gscale=_gs2.selectbox('Continuous scale',['Viridis','Turbo','Cividis','RdBu_r'],index=0,key=f'{prefix}_disp_graph_scale')
        _gssym=_gs3.selectbox('Sample symbol',['circle','square','diamond','cross','x','triangle-up','triangle-down','star'],index=0,key=f'{prefix}_disp_graph_ssym')
        _sumsym=_gs4.selectbox('Summary symbol',['diamond','star','square','circle','cross','x','triangle-up'],index=0,key=f'{prefix}_disp_graph_sumsym')
        _gs5,_gs6,_gs7,_gs8=st.columns(4)
        _sumcm=_gs5.selectbox('Summary colour',['Match groups','Fixed'],index=0,key=f'{prefix}_disp_graph_sumcm')
        _sumc=_gs6.color_picker('Fixed summary colour','#f5bf42',key=f'{prefix}_disp_graph_sumc')
        _gssz=typed_slider(_gs7,'Sample size',3,14,6,1,key=f'{prefix}_disp_graph_ssz')
        _gsopac=typed_slider(_gs8,'Sample opacity',10,100,30,5,key=f'{prefix}_disp_graph_sopac')
    _ghov2=hover_cols(_fm,['Sample_ID','Grouping_Method','Group_ID','Age_Ma','Dataset','Arc_or_Segment','Geologic_Domain','Rock_Type_Model','Model','Predicted_km','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb']+[p for p in PROXY_THICKNESS_LABELS if p in _fm],_gx,_gy,_gc)
    _pgfig=go.Figure()
    _sd=plot_df(tidy_numbers(_fm.copy()))
    _sd[_gx]=numeric_series(_sd,_gx); _sd[_gy]=numeric_series(_sd,_gy)
    _sd=_sd.replace([np.inf,-np.inf],np.nan).dropna(subset=[_gx,_gy]).copy()
    _ddc=[c for c in ['Grouping_Method','Group_ID','Sample_ID',_gx,_gy] if c in _sd]
    if _ddc: _sd=_sd.drop_duplicates(subset=_ddc)
    if _shgsamp and not _sd.empty:
        _gcnum=_gc in _sd and pd.api.types.is_numeric_dtype(_sd[_gc])
        _sfig=px.scatter(_sd,x=_gx,y=_gy,color=_gc,hover_data=map_hover_data(_sd,_ghov2),template='plotly_white',labels={_gx:local_option_label(_gx),_gy:local_option_label(_gy),_gc:local_option_label(_gc)},color_discrete_sequence=qualitative_palette(_gpal),color_continuous_scale=_gscale if _gcnum else None)
        for _tr in _sfig.data:
            _tr.name=f'samples: {_tr.name}'; _tr.showlegend=_shgleg; _tr.marker.size=_gssz; _tr.marker.symbol=_gssym; _tr.marker.line=dict(color='black',width=0.35); _tr.opacity=_gsopac/100
            _pgfig.add_trace(_tr)
    _mrows=[]
    if _gstat!='Off' and not _sd.empty and {'Grouping_Method','Group_ID'}.issubset(_sd):
        for (_meth,_grpid),_g in _sd.groupby(['Grouping_Method','Group_ID'],dropna=False):
            _gv=_g.replace([np.inf,-np.inf],np.nan).dropna(subset=[_gx,_gy]).copy()
            if len(_gv)<int(_gminn): continue
            _xv=pd.to_numeric(_gv[_gx],errors='coerce').dropna(); _yv=pd.to_numeric(_gv[_gy],errors='coerce').dropna()
            if _xv.empty or _yv.empty: continue
            _xs=float(_xv.mean() if _gstat=='Mean' else _xv.median())
            _ys=float(_yv.mean() if _gstat=='Mean' else _yv.median())
            _mrows.append({'Grouping_Method':_meth,'Group_ID':_grpid,_gx:_xs,_gy:_ys,'n':len(_gv),'X_Q25':float(_xv.quantile(0.25)),'X_Q75':float(_xv.quantile(0.75)),'Y_Q25':float(_yv.quantile(0.25)),'Y_Q75':float(_yv.quantile(0.75))})
    _mdf=pd.DataFrame(_mrows).dropna(subset=[_gx,_gy]) if _mrows else pd.DataFrame()
    if not _mdf.empty:
        _mdf['_EX_plus']=(_mdf['X_Q75']-_mdf[_gx]).clip(lower=0); _mdf['_EX_minus']=(_mdf[_gx]-_mdf['X_Q25']).clip(lower=0)
        _mdf['_EY_plus']=(_mdf['Y_Q75']-_mdf[_gy]).clip(lower=0); _mdf['_EY_minus']=(_mdf[_gy]-_mdf['Y_Q25']).clip(lower=0)
        _mfig=px.scatter(_mdf,x=_gx,y=_gy,color='Group_ID',symbol='Grouping_Method',hover_data=['Grouping_Method','Group_ID','n','X_Q25','X_Q75','Y_Q25','Y_Q75'],template='plotly_white',labels={_gx:local_option_label(_gx),_gy:local_option_label(_gy),'Group_ID':f'group {_gstat.lower()}'},color_discrete_sequence=qualitative_palette(_gpal),error_x='_EX_plus',error_x_minus='_EX_minus',error_y='_EY_plus',error_y_minus='_EY_minus')
        for _tr in _mfig.data:
            _tr.name=f'{_gstat.lower()}: {_tr.name}'; _tr.showlegend=_shgleg; _tr.marker.size=13; _tr.marker.symbol=_sumsym
            if _sumcm=='Fixed': _tr.marker.color=_sumc
            _tr.marker.line=dict(color='black',width=1.1)
            _pgfig.add_trace(_tr)
    if _gstat!='Off' and _mdf.empty:
        st.info(f'No {_gstat.lower()} markers meet the selected X/Y fields and minimum N.')
    _pgfig.update_layout(height=430,margin=dict(l=10,r=10,t=10,b=10),showlegend=_shgleg)
    _pgfig.update_xaxes(title=local_option_label(_gx)); _pgfig.update_yaxes(title=local_option_label(_gy))
    st.plotly_chart(_pgfig,width='stretch',key=f'{prefix}_disp_graph_fig')

@st.fragment
def _render_feature_importance_panel(importance_df, models_keys, model_key, sort_key):
    """Feature importance picker + figure, isolated in a fragment so the
    model / sort selectboxes don't trigger a full-tab rerun that would
    rebuild every other chart on the tab."""
    if importance_df.empty:
        st.info('Feature importance appears after the model is trained.')
        return
    fi1, fi2 = st.columns([1, 0.7])
    _fi_combined = 'All models — combined (mean bars + per-model dots)'
    _fi_options = [_fi_combined] + list(models_keys)
    normalize_widget_state(model_key, _fi_options)
    _model_for_importance = fi1.selectbox('Feature importance model', _fi_options, key=model_key)
    _importance_sort      = fi2.selectbox('Sort features by', ['Importance', 'Compatibility'], index=0, key=sort_key)
    if _model_for_importance == _fi_combined:
        fig = combined_feature_weighting_figure(importance_df, sort_by=_importance_sort, height=360)
        st.caption(
            'Bars show the **mean** Relative_Importance across every trained '
            "model. Each coloured dot is one model's individual score for that "
            'feature — clustered dots = models agree, scattered dots = '
            'models disagree. With signed importance, dots above the zero '
            'line mean *high values raise the prediction*; below = *high '
            'values lower it*.'
        )
    else:
        _top = (importance_df[importance_df['Model'].eq(_model_for_importance)]
                .sort_values('Relative_Importance', ascending=False))
        fig = feature_weighting_figure(_top['Feature'].tolist(), _top, sort_by=_importance_sort, height=360)
    if fig is not None:
        st.plotly_chart(fig, width='stretch')

@st.fragment
def _render_blind_validation_map(test_bench):
    """Validate-tab Blind validation map, isolated in a fragment so its
    controls (layer selectbox, zoom checkbox) don't trigger a full-tab
    rerun that would rebuild every other Validate chart."""
    if not {'Lat','Lon'}.issubset(test_bench):
        st.info('Validation map needs Lat and Lon columns.')
        return
    map_options=[c for c in ['Predicted_km','Observed_km','Residual_km','CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Sediment_km','Age_Ma','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type_Model'] if c in test_bench]
    if not map_options:
        st.info('No validation columns are available for map colouring.')
        return
    map_c1,map_c2,map_c3=st.columns([1,1,0.7])
    map_c1.markdown('**crust 1.0: crustal thickness [Km]**')
    normalize_widget_state('validation_map_layer',map_options)
    map_color=map_c2.selectbox('Validation layer',map_options,index=0,key='validation_map_layer',format_func=lambda c: 'model: crustal thickness [Km]' if c == 'Predicted_km' else 'known: crustal thickness [Km]' if c == 'Observed_km' else 'model: residual [Km]' if c == 'Residual_km' else model_map_option_label(c,0))
    validation_zoom_to_data=map_c3.checkbox('Zoom to data',value=False,key='validation_map_zoom_to_data')
    m=plot_df(tidy_numbers(test_bench).dropna(subset=['Lat','Lon']))
    m_geo = geo_downsample(m)
    if len(m_geo) < len(m): map_c3.caption(f'{len(m_geo):,} / {len(m):,} pts')
    hover_columns=hover_cols(m_geo,['Sample_ID','Age_Ma','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type_Model','Observed_km','Predicted_km','Residual_km','CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Sediment_km','CRUST1_Match_Distance_km','Model'],map_color,'Lat','Lon')
    fig=px.scatter_geo(
        m_geo,lat='Lat',lon='Lon',color=map_color,
        hover_data=map_hover_data(m_geo,hover_columns),
        projection='natural earth',template='plotly_white',
        labels={map_color:model_map_legend_title(map_color).replace('<br>',' ')}
    )
    fig.update_traces(marker=dict(line=dict(color='black',width=0.7)),selector=dict(type='scattergeo'))
    default_crust=read_default_crust_grid(str(DEFAULT_CRUST1_GRID))
    if not default_crust.empty:
        fig=add_crust_grid_background_to_geofig(fig,default_crust,'CRUST1_Total_Crust_km')
    fig=style_training_map_legends(fig,map_color)
    fig=apply_geo_zoom_to_data(fig,m_geo,validation_zoom_to_data)
    fig.update_layout(height=650)
    st.plotly_chart(fig,width='stretch')

t_data_prep,t_grouping,t0,t_validation,t_unknown,t_result_summary=st.tabs(['Prepare','Group','Model','Validate','Predict','Summary'])

models={}; validation_df=pd.DataFrame(); importance_df=pd.DataFrame(); train_df=pd.DataFrame(); clean=pd.DataFrame(); target=None
model_options=[]  # populated by ML tab when trained; empty in ratio/GAME mode
selected_model_names=['Guo & Yang (2023)']; selected_algorithms=['ExtraTrees']; feature_set_name='Guo & Yang (2023)'; selected_features=FEATURE_SETS[feature_set_name]; model=None

with t_data_prep:
    st.header('Data Preparation')
    st.caption('Upload files, confirm column mapping, assign each file to one or more tabs — then use them anywhere.')
    st.markdown(
        '<div style="display:flex;gap:18px;margin:4px 0 14px 0;font-size:12px;font-family:sans-serif">'
        '<span><span style="display:inline-block;width:12px;height:12px;background:#fef2f2;border:1px solid #dc2626;border-radius:2px;margin-right:4px"></span>'
        '<span style="color:#dc2626;font-weight:600">Red</span> = auto-detected, not yet confirmed</span>'
        '<span><span style="display:inline-block;width:12px;height:12px;background:#eff6ff;border:1px solid #1d4ed8;border-radius:2px;margin-right:4px"></span>'
        '<span style="color:#1d4ed8;font-weight:600">Blue</span> = confirmed</span>'
        '<span><span style="display:inline-block;width:12px;height:12px;background:#f3f4f6;border:1px solid #9ca3af;border-radius:2px;margin-right:4px"></span>'
        '<span style="color:#6b7280;font-weight:600">Grey</span> = no mapping found (kept as-is)</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Session reset ────────────────────────────────────────────────────────
    # Clears in-session state (pool, role assignments, applied maps, grouping
    # state, multi-model slots, cached file reads) and forces both file
    # uploaders to re-instantiate with fresh keys so dropped files actually
    # disappear from the UI.
    #
    # PRESERVED on reset (so saved configuration survives):
    #   • _dp_user_aliases.json   — column-mapping decisions
    #   • _dp_default_refs.json   — auto-load list (Guo & Yang etc. come back)
    #   • _dp_custom_refs.json    — registered custom files
    #   • _dp_custom_refs_uploads/ — uploaded custom-ref files on disk
    # Saved references reseed _dp_ref_loaded on the rerun and reappear in the
    # pool. Drag-dropped files (e.g. Luffi as a one-off upload) DO disappear
    # because the file_uploader gets a new key.
    _rst_c1, _rst_c2 = st.columns([5, 1])
    with _rst_c2:
        _rst_reset_aliases = st.checkbox(
            'Reset mappings too',
            value=False,
            key='dp_reset_aliases',
            help='Also wipe saved column-mapping decisions. '
                 'Next upload will re-ask for all column assignments.',
        )
        if st.button('🗑 Reset session', key='dp_reset_session', use_container_width=True,
                     help='Clear current uploads, the in-session pool, role assignments, '
                          'and all grouping/model state. Saved references (Guo etc.) are '
                          'preserved; tick "Reset mappings too" to also wipe column aliases.'):
            # Capture the file-uploader nonces BEFORE the wipe loop pops them,
            # so we can strictly increment (each Reset advances to a key that
            # has never been used before — guarantees no stale file state).
            _pre_uploader_nonce     = st.session_state.get('_dp_uploader_nonce', 0)
            _pre_crf_uploader_nonce = st.session_state.get('_dp_crf_uploader_nonce', 0)
            # Wipe pool + role assignments
            for _k in [
                'prepped_datasets', '_dp_dataset_roles', '_dp_sheet_pool_keys',
                '_dp_ref_loaded',
                'dp_training_df', 'dp_validation_df', 'dp_prediction_df',
                '_g_apply_preview', '_g_med_preview', '_g_med_name', '_g_bench_sig',
                '_rs_pred_bench', '_rs_group_map', '_training_sources_used',
                '_cached_importance_df', '_mm_slots',
            ]:
                st.session_state.pop(_k, None)
            # Wipe per-file state from previous uploads — anything matching the
            # standard prefixes used during the Prepare tab's processing.
            _stale_prefixes = ('dp_', '_dp_', '_g_', 'mm_', 'predict_', 'val_', 'primary_')
            for _k in list(st.session_state.keys()):
                if any(_k.startswith(_p) for _p in _stale_prefixes):
                    st.session_state.pop(_k, None)
            # Bump the file-uploader nonces so Streamlit re-creates the
            # widgets with fresh keys — popping their session-state entries
            # alone doesn't clear the dropped files. Strict increment from
            # the pre-wipe values ensures repeated Resets keep advancing.
            st.session_state['_dp_uploader_nonce']     = _pre_uploader_nonce + 1
            st.session_state['_dp_crf_uploader_nonce'] = _pre_crf_uploader_nonce + 1
            # Clear file-read caches too
            try:
                read_table.clear()
            except Exception:
                pass
            try:
                enrich.clear()
            except Exception:
                pass
            # Optionally wipe column-mapping alias file
            if st.session_state.get('dp_reset_aliases', False):
                try:
                    _DP_USER_ALIAS_FILE.write_text('{}', encoding='utf-8')
                except Exception:
                    pass
            st.rerun()

    # File-uploader widgets retain their files internally regardless of
    # session-state manipulation — popping the key isn't enough to clear
    # them. The standard Streamlit pattern is to bump a counter that's part
    # of the widget key so the next render creates a fresh widget instance
    # with no files. The Reset Session handler increments this counter.
    _dp_uploader_nonce = st.session_state.setdefault('_dp_uploader_nonce', 0)
    _dp_uploads = st.file_uploader(
        'Upload file(s)',
        type=['csv','xlsx','xls'],
        accept_multiple_files=True,
        key=f'dp_global_uploader__{_dp_uploader_nonce}',
        help='Upload any number of files — training, validation, prediction, or mixed. Assign each below.',
    )

    # ── Load a built-in reference dataset through Prepare ────────────────────
    # The default Model-tab path loads Guo / Zou / Luffi calibration directly
    # from disk and bypasses Prepare. That works fine when the validation file
    # also has canonical column names — but when your validation file uses
    # variants like `SiO2 (wt%)` instead of `SiO2`, both training and
    # validation need to flow through the same column-mapping pipeline so
    # they end up aligned. These buttons load the reference file as if you
    # uploaded it: it appears below alongside any user uploads, with the
    # same mapping editor, the same role assignment, and the same applied-map
    # processing.
    import io as _io
    def _make_ref_file_like(name, path):
        """Build a file-like object that pd.read_excel and openpyxl will accept.

        BytesIO already implements the full IO protocol (seekable, readable,
        tell, getbuffer, etc.), so we just attach a .name attribute on top —
        much simpler than rolling our own wrapper class.
        """
        bio = _io.BytesIO(Path(path).read_bytes())
        bio.name = name
        return bio

    # Define the reference options once — used for both auto-load and the
    # button rendering below. Built-ins are checked for an actual file on
    # disk; missing ones are dropped silently so the loader only shows
    # references that can actually be loaded. User-registered custom refs
    # (saved to _dp_custom_refs.json) are appended.
    _ref_options = [
        (_label, _path) for (_label, _path) in [
            ('Guo & Yang (2023)',                     find_training()),
            ('Zou et al. (2021)',                     find_zou_training()),
            ('Luffi & Ducea (2022) — calibration',    find_luffi_training()),
        ]
        if _path is not None
    ]
    _builtin_ref_labels = {l for (l, _p) in _ref_options}
    for _cr in _dp_load_custom_refs():
        # Skip if a custom entry collides with a built-in label or has no file
        if _cr['label'] in _builtin_ref_labels:
            continue
        _cr_path = Path(_cr['path']) if _cr.get('path') else None
        if _cr_path and _cr_path.exists():
            _ref_options.append((_cr['label'], _cr_path))
        # Files missing on disk are not included as buttons; they still show
        # in the "Registered custom references" list with a ⚠ flag, so the
        # user can remove them.
    _ref_label_to_path = {l: p for (l, p) in _ref_options}

    # First-time init: try to auto-load whatever the user saved last time as
    # the default preload list. The actual file-read + mapping pipeline runs
    # naturally as soon as these entries appear in `_dp_ref_loaded` because
    # the synthetic-uploads loop below picks them up. Column-mapping decisions
    # carry over via `_dp_user_aliases.json` (auto-saved per change).
    if '_dp_ref_loaded' not in st.session_state:
        _saved_default_labels = _dp_load_default_refs()
        st.session_state['_dp_ref_loaded'] = [
            (l, str(_ref_label_to_path[l]))
            for l in _saved_default_labels
            if l in _ref_label_to_path and _ref_label_to_path[l] is not None
        ]

    with st.expander('📚 Load a reference dataset through Prepare (Guo / Zou / Luffi)', expanded=False):
        st.caption(
            'Run a built-in calibration dataset through the same column-mapping '
            'pipeline as your uploads. Use this when training and validation files '
            'use different column-name conventions and you need them aligned — '
            'e.g. validating a Guo-trained model against a Luffi-style file with '
            '`SiO2 (wt%)` column names.'
        )
        if not _ref_options:
            st.caption(
                '_No reference files found beside the app. Register a custom '
                'one below by dragging a csv/xlsx file._'
            )
        else:
            _ref_cols = st.columns(len(_ref_options))
            _already = {label for (label, _path) in st.session_state['_dp_ref_loaded']}
            for _ri, ((_ref_label, _ref_path), _ref_col) in enumerate(zip(_ref_options, _ref_cols)):
                with _ref_col:
                    if _ref_label in _already:
                        if st.button(f'✓ {_ref_label} loaded', key=f'dp_ref_unload_{_ri}',
                                     use_container_width=True,
                                     help='Click to remove this reference dataset from Prepare.'):
                            st.session_state['_dp_ref_loaded'] = [
                                (l, p) for (l, p) in st.session_state['_dp_ref_loaded']
                                if l != _ref_label
                            ]
                            st.rerun()
                    else:
                        if st.button(f'+ {_ref_label}', key=f'dp_ref_load_{_ri}',
                                     use_container_width=True,
                                     help=f'Load {_ref_label} from disk and route it through Prepare.'):
                            # Defensive: re-check session state at click time so a
                            # rapid double-click (before Streamlit reruns and swaps
                            # the button to the unload variant) can't double-add.
                            _cur_loaded = st.session_state['_dp_ref_loaded']
                            if not any(_l == _ref_label for (_l, _) in _cur_loaded):
                                _cur_loaded.append((_ref_label, str(_ref_path)))
                            st.rerun()

        # ── Register a custom reference file ─────────────────────────────
        # Lets the user pin any csv/xlsx on disk so it appears as a
        # one-click button alongside Guo/Zou/Luffi. Entries persist across
        # sessions via _dp_custom_refs.json.
        st.divider()
        with st.expander('➕ Register a custom reference file', expanded=False):
            st.caption(
                'Drop a csv/xlsx file here to pin it as a one-click reference. '
                'The file is copied into the app folder (`_dp_custom_refs_uploads/`) '
                'so it survives across sessions even if the source moves. After '
                'registering, click its **+** button above to load it through '
                'Prepare, set up the column mapping, then **💾 Save as default '
                'preload** to make it auto-load on every future session.'
            )
            _custom_existing = _dp_load_custom_refs()
            _custom_existing_labels = {e['label'] for e in _custom_existing}

            # Folder where dropped reference files are copied. Created on demand;
            # the path itself is what gets stored in _dp_custom_refs.json.
            _DP_CUSTOM_UPLOADS_DIR = _APP_DIR / '_dp_custom_refs_uploads'

            # Drag-and-drop file uploader + friendly-name input.
            # Re-key with a nonce so Reset Session can clear it (file_uploader
            # otherwise retains the dropped file even when session-state is popped).
            _crf_uploader_nonce = st.session_state.setdefault('_dp_crf_uploader_nonce', 0)
            _crf_file = st.file_uploader(
                'Drop a csv/xlsx file here',
                type=['csv', 'xlsx', 'xls'],
                key=f'_dp_custom_ref_uploader__{_crf_uploader_nonce}',
                help='The file is copied into _dp_custom_refs_uploads/ on click.',
            )
            _crf_c1, _crf_c2 = st.columns([4, 1])
            _crf_label_input = _crf_c1.text_input(
                'Friendly name',
                key='_dp_custom_ref_new_label',
                placeholder=(_crf_file.name.rsplit('.', 1)[0]
                             if _crf_file is not None else 'My calibration set'),
            )
            # Default the friendly name to the filename stem if the user hasn't
            # typed anything yet — saves a step in the common case.
            _crf_label_effective = (
                _crf_label_input.strip() or
                (_crf_file.name.rsplit('.', 1)[0] if _crf_file is not None else '')
            )
            _crf_can_add = (
                _crf_file is not None
                and bool(_crf_label_effective)
                and _crf_label_effective not in _builtin_ref_labels
                and _crf_label_effective not in _custom_existing_labels
            )
            with _crf_c2:
                st.write('')   # vertical spacer to align with the text input
                if st.button('+ Register', key='_dp_custom_ref_add',
                             use_container_width=True, disabled=not _crf_can_add,
                             help='Save the dropped file under the app folder and '
                                  'add it to the reference loader.'):
                    try:
                        _DP_CUSTOM_UPLOADS_DIR.mkdir(exist_ok=True)
                        # Sanitise the filename so it's safe across OSes —
                        # keep alphanumerics, dot, dash, underscore.
                        _safe_name = re.sub(
                            r'[^\w.\-]+', '_',
                            _crf_file.name.strip()
                        )
                        _target_path = _DP_CUSTOM_UPLOADS_DIR / _safe_name
                        # If a file with the same name exists, suffix with a
                        # counter so we never silently overwrite something.
                        _stem, _suffix = _target_path.stem, _target_path.suffix
                        _ctr = 1
                        while _target_path.exists():
                            _target_path = _DP_CUSTOM_UPLOADS_DIR / f'{_stem}_{_ctr}{_suffix}'
                            _ctr += 1
                        _target_path.write_bytes(_crf_file.getvalue())
                        _custom_existing.append({
                            'label': _crf_label_effective,
                            'path':  str(_target_path.resolve()),
                        })
                        if _dp_save_custom_refs(_custom_existing):
                            # Clear the inputs so the form is fresh — pop the
                            # text input, and bump the uploader nonce so the
                            # file_uploader gets a fresh widget instance.
                            st.session_state.pop('_dp_custom_ref_new_label', None)
                            st.session_state['_dp_crf_uploader_nonce'] = (
                                st.session_state.get('_dp_crf_uploader_nonce', 0) + 1
                            )
                            st.rerun()
                        else:
                            st.error(
                                f'Could not write to {_DP_CUSTOM_REFS_FILE}. '
                                'Check that the app directory is writable.'
                            )
                    except Exception as _crf_err:
                        st.error(f'Could not register file: {_crf_err}')

            # Validation hints — only show when there's actually input to react to
            if _crf_label_effective and _crf_label_effective in _builtin_ref_labels:
                st.warning(f'"{_crf_label_effective}" is already a built-in reference name. Pick a different label.')
            elif _crf_label_effective and _crf_label_effective in _custom_existing_labels:
                st.warning(f'"{_crf_label_effective}" is already registered. Remove it below first if you want to re-add it.')

            # List existing custom entries
            if _custom_existing:
                st.markdown('**Registered custom references:**')
                _to_remove = None
                for _cri, _cr in enumerate(_custom_existing):
                    _r1, _r2, _r3 = st.columns([1.5, 3.5, 1])
                    _r1.markdown(f'**{_cr["label"]}**')
                    _missing = not Path(_cr['path']).exists()
                    _r2.code(_cr['path'] + ('  ⚠ file missing' if _missing else ''),
                             language=None)
                    if _r3.button('🗑 Remove', key=f'_dp_custom_ref_del_{_cri}',
                                  use_container_width=True,
                                  help='Unregister this custom reference. The file on disk is not deleted.'):
                        _to_remove = _cri
                if _to_remove is not None:
                    _removed = _custom_existing[_to_remove]
                    _removed_label = _removed['label']
                    _removed_path  = Path(_removed['path'])
                    del _custom_existing[_to_remove]
                    _dp_save_custom_refs(_custom_existing)
                    # Cascade: drop the removed label from the in-session
                    # loaded list and the saved default-preload JSON, so it
                    # doesn't try to re-load a now-unregistered file.
                    st.session_state['_dp_ref_loaded'] = [
                        (l, p) for (l, p) in st.session_state.get('_dp_ref_loaded', [])
                        if l != _removed_label
                    ]
                    _saved_defaults = _dp_load_default_refs()
                    if _removed_label in _saved_defaults:
                        _dp_save_default_refs(
                            [l for l in _saved_defaults if l != _removed_label]
                        )
                    # If the file lives inside our managed uploads folder,
                    # delete it so we don't accumulate orphans. External paths
                    # the user pinned (registered before drag-and-drop, or
                    # edited the JSON manually) are NOT deleted — they could
                    # be on a network share the user doesn't want touched.
                    try:
                        if (_removed_path.exists()
                                and _removed_path.parent.resolve() == (_APP_DIR / '_dp_custom_refs_uploads').resolve()):
                            _removed_path.unlink()
                    except Exception:
                        pass
                    st.rerun()
            else:
                st.caption('_No custom references registered yet._')

        # ── Save current selection as the default preload ────────────────
        # Once the user has clicked + on the references they want and run them
        # through the mapping editor (mappings auto-persist via
        # _dp_user_aliases.json), this button writes the *list* of loaded
        # references to disk so they reappear on every future session.
        st.divider()
        _saved_now = _dp_load_default_refs()
        _current_labels = [l for (l, _) in st.session_state['_dp_ref_loaded']]
        _save_c1, _save_c2 = st.columns([3, 2])
        with _save_c1:
            if _saved_now:
                st.caption(
                    f'**Currently auto-loaded on startup:** {", ".join(_saved_now)}. '
                    'Click below to overwrite with your current selection.'
                    if _saved_now != _current_labels else
                    f'**Currently auto-loaded on startup:** {", ".join(_saved_now)} ✓ matches.'
                )
            else:
                st.caption(
                    'No defaults saved yet. Click below to make the references '
                    'currently loaded in this session reappear automatically '
                    'on every future startup.'
                )
        with _save_c2:
            _save_disabled = (_saved_now == _current_labels)
            if st.button(
                '💾 Save as default preload',
                key='dp_ref_save_defaults',
                use_container_width=True,
                disabled=_save_disabled,
                help='Overwrite _dp_default_refs.json with the references currently '
                     'loaded above. Column-mapping decisions are saved separately '
                     'in _dp_user_aliases.json (auto-saved on every change).',
            ):
                if _dp_save_default_refs(_current_labels):
                    if _current_labels:
                        st.success(
                            f'Saved {len(_current_labels)} reference dataset(s) as default. '
                            'They\'ll auto-load on every future session.'
                        )
                    else:
                        st.success('Cleared default preload list.')
                    st.rerun()
                else:
                    st.error(
                        f'Could not write to {_DP_DEFAULT_REFS_FILE}. '
                        'Check that the app directory is writable.'
                    )

    # Build the synthetic uploads list from the loaded reference datasets
    _dp_synthetic_uploads = []
    for _ref_label, _ref_path_str in st.session_state.get('_dp_ref_loaded', []):
        _p = Path(_ref_path_str)
        if not _p.exists():
            continue
        # Use a stable display name so the Prepare-tab session-state keys
        # (which include _dp_f.name) survive across reruns for this entry.
        try:
            _dp_synthetic_uploads.append(_make_ref_file_like(_p.name, str(_p)))
        except Exception:
            # Skip files we can't read (permissions, locked, etc.)
            pass

    # Combined list — synthetic refs go AFTER user uploads so user files keep
    # their lower indices (and their session-state keys) when refs are added.
    _dp_uploads_combined = list(_dp_uploads or []) + _dp_synthetic_uploads

    # Buckets filled per file; merged and stored at the end
    _dp_dest_dfs: dict = {'dp_training_df': [], 'dp_validation_df': [], 'dp_prediction_df': []}
    _dp_key = 'dp'   # single namespace for session-state keys
    _dp_individual_sheets: dict = {}   # filename-stem → processed DataFrame
    _dp_individual_roles:  dict = {}   # filename-stem → set of role strings

    for _dp_fi, _dp_f in enumerate(_dp_uploads_combined):
        st.divider()

        # ── Sheet selector (Excel only) ──────────────────────────────────────
        _dp_sheet_key = f'{_dp_key}_sheet_{_dp_fi}_{_dp_f.name}'
        _dp_is_excel = _dp_f.name.lower().endswith(('.xlsx', '.xls'))
        if _dp_is_excel:
            _dp_f.seek(0)
            _dp_sheets = _dp_excel_sheet_names(_dp_f)
            if len(_dp_sheets) > 1:
                _dp_sheet_default = st.session_state.get(_dp_sheet_key, _dp_sheets[0])
                if _dp_sheet_default not in _dp_sheets:
                    _dp_sheet_default = _dp_sheets[0]
                _dp_sheet_sel = st.selectbox(
                    f'Sheet — {_dp_f.name}', _dp_sheets,
                    index=_dp_sheets.index(_dp_sheet_default),
                    key=_dp_sheet_key,
                )
                if st.session_state.get(_dp_sheet_key) != _dp_sheet_sel:
                    # Sheet changed → force remap and clear applied snapshot
                    _remap_key = f'{_dp_key}_map_{_dp_fi}_{_dp_f.name}'
                    st.session_state.pop(_remap_key, None)
                    st.session_state.pop(f'{_dp_key}_applied_map_{_dp_fi}_{_dp_f.name}', None)
                    st.session_state.pop(f'{_dp_key}_applied_type_{_dp_fi}_{_dp_f.name}', None)
            else:
                _dp_sheet_sel = _dp_sheets[0] if _dp_sheets else 0
        else:
            _dp_sheet_sel = 0

        # Cache raw read in session state — avoids re-reading the file on every widget interaction
        _dp_raw_key = f'{_dp_key}_raw_{_dp_fi}_{_dp_f.name}_{_dp_sheet_sel}'
        if _dp_raw_key not in st.session_state:
            _dp_f.seek(0)
            _raw_read, _orig_read = _dp_read_raw(_dp_f, sheet_name=_dp_sheet_sel)
            st.session_state[_dp_raw_key] = (_raw_read, _orig_read)
        _dp_raw, _dp_orig = st.session_state[_dp_raw_key]

        # ── Collapsible card: collapse once all columns confirmed ────────────
        _dp_card_key = f'{_dp_key}_card_{_dp_fi}_{_dp_f.name}'
        _dp_mkey = f'{_dp_key}_map_{_dp_fi}_{_dp_f.name}'
        _dp_ckey = f'{_dp_key}_conf_{_dp_fi}_{_dp_f.name}'
        _dp_confirmed_already = st.session_state.get(_dp_ckey, set())
        _dp_all_confirmed = (
            bool(_dp_confirmed_already) and
            all(_dp_confirmed_already.__contains__(c) or
                st.session_state.get(_dp_mkey, {}).get(c, _DP_KEEP_ORIGINAL) == _DP_KEEP_ORIGINAL
                for c in _dp_orig)
        )
        if _dp_raw.empty:
            st.warning(f'Could not read {_dp_f.name}')
            continue

        # ── Phase gate: show raw preview until user clicks Auto-map ─────────
        # The gate exists so a 50-MB drag-drop doesn't burn cycles auto-mapping
        # something the user might immediately remove.  But for synthetic
        # uploads (saved-default references / "+" reference-loader clicks),
        # the user has already explicitly asked for the file — and on session
        # restart, the gate would prevent the pool from being populated until
        # the user manually visits Prepare and clicks "Auto-map columns".  We
        # auto-skip the gate for those, so saved defaults flow through to the
        # pool with zero clicks each session.
        _is_synthetic_upload = _dp_fi >= len(_dp_uploads or [])
        _dp_started_key = f'{_dp_key}_started_{_dp_fi}_{_dp_f.name}'
        # Already mapped if mapping keys exist in session state from a previous run
        if (
            (_dp_mkey in st.session_state and _dp_started_key not in st.session_state)
            or (_is_synthetic_upload and _dp_started_key not in st.session_state)
        ):
            st.session_state[_dp_started_key] = True
        _dp_mapping_started = st.session_state.get(_dp_started_key, False)

        _dp_card_expanded = st.session_state.get(_dp_card_key, not _dp_all_confirmed)
        with st.expander(f'**File {_dp_fi+1} — {_dp_f.name}**' + (' ✓' if _dp_all_confirmed else ''), expanded=_dp_card_expanded):

            if not _dp_mapping_started:
                # ── Phase 1: raw preview only — fast ────────────────────────
                st.caption(f'{len(_dp_raw):,} rows × {len(_dp_orig)} columns')
                st.dataframe(
                    _dp_raw.head(8).astype(str).replace('nan', ''),
                    use_container_width=True, hide_index=True,
                )
                _ph1, _ph2, _ph3 = st.columns([2, 1, 1])
                if _ph1.button(
                    '🔍 Auto-map columns',
                    key=f'{_dp_key}_automap_btn_{_dp_fi}_{_dp_f.name}',
                    type='primary',
                    help='Detect column mappings automatically from header names',
                ):
                    st.session_state[_dp_started_key] = True
                    st.rerun()
                _ph2.caption(f'{len(_dp_orig)} columns detected')
                # Skip the rest of the per-file block until mapping is started
                continue

            # ── Phase 2: full mapping editor (only after Auto-map clicked) ──
            # Per-file session-state keys
            _dp_tkey  = f'{_dp_key}_type_{_dp_fi}_{_dp_f.name}'
            _dp_ukey  = f'{_dp_key}_unit_{_dp_fi}_{_dp_f.name}'
            # Validate session state — reset if stale (old internal-name format
            # OR migration from pre-bare-label registry)
            if (_dp_mkey not in st.session_state or
                    any(v not in _DP_DISPLAY_LABELS for v in st.session_state[_dp_mkey].values())):
                _auto_map_result, _auto_unit_result = _dp_auto_map(_dp_orig)
                st.session_state[_dp_mkey] = _auto_map_result
                st.session_state[_dp_ukey] = _auto_unit_result
                st.session_state.pop(_dp_tkey, None)   # force retype on map reset
            if _dp_ukey not in st.session_state:
                # Mapping survived from a previous run but the unit map didn't —
                # rebuild units from the existing display labels.
                _existing_map = st.session_state[_dp_mkey]
                _rebuilt_units = {}
                for _oc, _dl in _existing_map.items():
                    _detected = _dp_detect_unit(_oc)
                    if _dl != _DP_KEEP_ORIGINAL and _dl in _DP_DISPLAY_TO_INTERNAL:
                        _, _du, _ = _DP_DISPLAY_TO_INTERNAL[_dl]
                        _rebuilt_units[_oc] = _detected or _du
                    else:
                        _rebuilt_units[_oc] = _detected
                st.session_state[_dp_ukey] = _rebuilt_units
            if _dp_ckey not in st.session_state:
                st.session_state[_dp_ckey] = {c for c,m in st.session_state[_dp_mkey].items() if m == _DP_KEEP_ORIGINAL}
            _dp_cur_map   = st.session_state[_dp_mkey]
            _dp_cur_unit  = st.session_state[_dp_ukey]
            _dp_confirmed = st.session_state[_dp_ckey]
            # Column-type state: auto-detect on first load only
            _auto_ignored_log_key = f'{_dp_tkey}_auto_ignored'
            if _dp_tkey not in st.session_state:
                _auto_types = _dp_auto_type(_dp_orig, _dp_cur_map, _dp_raw)
                # Auto-Ignore second+ occurrence of any duplicated mapped name.
                # We log every auto-Ignore (column → kept partner) into session
                # state so the UI can surface a notification — silent column
                # drops were the root cause of the FeOt/FEOwt vs FEOTwt iron
                # collision that produced wrong total-iron values.
                _seen_mapped_dl: dict = {}   # display_label → first orig col
                _auto_ignored_log: list = []
                for _oc in _dp_orig:   # preserve column order so first wins
                    _dl_oc = _dp_cur_map.get(_oc, _DP_KEEP_ORIGINAL)
                    if _dl_oc != _DP_KEEP_ORIGINAL:
                        if _dl_oc in _seen_mapped_dl:
                            _auto_types[_oc] = 'Ignore'
                            _auto_ignored_log.append({
                                'ignored_col':  _oc,
                                'kept_col':     _seen_mapped_dl[_dl_oc],
                                'shared_label': _dl_oc,
                            })
                        else:
                            _seen_mapped_dl[_dl_oc] = _oc
                st.session_state[_dp_tkey] = _auto_types
                st.session_state[_auto_ignored_log_key] = _auto_ignored_log
            _dp_cur_type: dict = st.session_state[_dp_tkey]

            # ── Applied (committed) snapshot — processing only uses these ─────
            _dp_applied_mkey = f'{_dp_key}_applied_map_{_dp_fi}_{_dp_f.name}'
            _dp_applied_tkey = f'{_dp_key}_applied_type_{_dp_fi}_{_dp_f.name}'
            _dp_applied_unit_key = f'{_dp_key}_applied_unit_{_dp_fi}_{_dp_f.name}'
            _dp_applied_rename_key = f'{_dp_key}_applied_rename_{_dp_fi}_{_dp_f.name}'
            _dp_rkey = f'{_dp_key}_rename_{_dp_fi}_{_dp_f.name}'
            # Validate the existing applied map against the current
            # _DP_DISPLAY_LABELS set. If any value is no longer recognised
            # (e.g. legacy labels like 'SiO2 [wt%]' from before the
            # bare-label refactor), reset applied state to the freshly
            # auto-mapped values. Without this, _dp_apply_mapping skips
            # renames it can't look up — silently producing a dataframe
            # with the original column names intact.
            _existing_applied = st.session_state.get(_dp_applied_mkey)
            _applied_is_stale = (
                _existing_applied is None
                or any(v not in _DP_DISPLAY_LABELS for v in _existing_applied.values())
            )
            if _applied_is_stale:
                st.session_state[_dp_applied_mkey] = dict(_dp_cur_map)
                st.session_state[_dp_applied_tkey] = dict(_dp_cur_type)
                st.session_state[_dp_applied_unit_key] = dict(_dp_cur_unit)
                st.session_state[_dp_applied_rename_key] = {}
            else:
                if _dp_applied_tkey not in st.session_state:
                    st.session_state[_dp_applied_tkey] = dict(_dp_cur_type)
                if _dp_applied_unit_key not in st.session_state:
                    st.session_state[_dp_applied_unit_key] = dict(_dp_cur_unit)
                if _dp_applied_rename_key not in st.session_state:
                    st.session_state[_dp_applied_rename_key] = {}
            if _dp_rkey not in st.session_state:
                st.session_state[_dp_rkey] = dict(st.session_state[_dp_applied_rename_key])
            _dp_applied_map:        dict = st.session_state[_dp_applied_mkey]
            _dp_applied_type:       dict = st.session_state[_dp_applied_tkey]
            _dp_applied_unit_map:   dict = st.session_state[_dp_applied_unit_key]
            _dp_applied_rename_map: dict = st.session_state[_dp_applied_rename_key]
            _dp_cur_rename:         dict = st.session_state[_dp_rkey]

            # Combined scrollable preview
            _dp_search_ph = f'dp_search_{_dp_fi}_{_dp_f.name}'
            st.markdown(_dp_preview_html(_dp_orig, _dp_cur_map, _dp_confirmed, _dp_raw, search_ph=_dp_search_ph), unsafe_allow_html=True)

            # Summary counts + mapping editor
            _dp_n_auto = sum(1 for c,m in _dp_cur_map.items() if m != _DP_KEEP_ORIGINAL and c not in _dp_confirmed)
            _dp_mc1, _dp_mc2 = st.columns([1,1])
            _dp_mc1.metric('Columns', len(_dp_orig))
            _dp_mc2.metric('Needs review', _dp_n_auto)

            # ── Mapping template import ──────────────────────────────────────
            with st.expander('Import mapping template', expanded=False):
                _tpl_up = st.file_uploader('Load mapping template (.json)',
                    type=['json'], key=f'{_dp_key}_tpl_up_{_dp_fi}_{_dp_f.name}')
                if _tpl_up is not None:
                    try:
                        _tpl_data = json.loads(_tpl_up.read().decode('utf-8'))
                        _tpl_map = {c: _tpl_data.get(c, _DP_KEEP_ORIGINAL) for c in _dp_orig}
                        _tpl_map = {c: v if v in _DP_DISPLAY_LABELS else _DP_KEEP_ORIGINAL for c, v in _tpl_map.items()}
                        st.session_state[_dp_mkey] = _tpl_map
                        st.session_state[_dp_applied_mkey] = dict(_tpl_map)
                        st.session_state[_dp_ckey] = set()
                        st.session_state.pop(_dp_tkey, None)
                        st.session_state.pop(_dp_applied_tkey, None)
                        st.success('Template applied — review and confirm mapping below.')
                        st.rerun()
                    except Exception as _te:
                        st.error(f'Could not read template: {_te}')
                st.divider()
                st.caption('Or re-import an edited Excel mapping file:')
                _xl_reimport = st.file_uploader('Load edited Excel mapping (.xlsx)',
                    type=['xlsx'], key=f'{_dp_key}_xl_reimport_{_dp_fi}_{_dp_f.name}')
                if _xl_reimport is not None:
                    try:
                        _xl_new_map, _xl_new_conf = _dp_excel_to_mapping(
                            _xl_reimport, _dp_orig, _dp_cur_map, _dp_confirmed)
                        st.session_state[_dp_mkey] = _xl_new_map
                        st.session_state[_dp_applied_mkey] = dict(_xl_new_map)
                        st.session_state[_dp_ckey] = _xl_new_conf
                        st.session_state.pop(_dp_tkey, None)
                        st.session_state.pop(_dp_applied_tkey, None)
                        st.success('Excel mapping imported.')
                        st.rerun()
                    except ValueError as _xe:
                        st.error(f'Could not read Excel mapping: {_xe}')

            # ── Mapping editor ───────────────────────────────────────────────
            # Architecture: the editor base DataFrame is always built from the
            # APPLIED (committed) snapshot so it never changes between Apply clicks.
            # The editor's own edited_rows layer shows the user's in-progress edits
            # on top without any rerun or base-data mutation → no scroll reset.
            # Nothing is written to session_state during editing; everything commits
            # on Apply.
            _editor_key  = f'{_dp_key}_editor_{_dp_fi}_{_dp_f.name}'
            _dp_edited   = pd.DataFrame()   # populated inside expander; readable outside
            _dp_has_pending = False          # updated inside expander; readable outside
            _dp_eff_map  = dict(_dp_applied_map)   # effective live state (for dup banner)
            _dp_eff_type = dict(_dp_applied_type)
            _dp_n_pending_map  = 0
            _dp_n_pending_type = 0

            with st.expander('Edit column mapping', expanded=(_dp_n_auto > 0)):
                _dp_filter_val = st.text_input(
                    '', key=f'{_dp_key}_filter_{_dp_fi}_{_dp_f.name}',
                    placeholder=_dp_search_ph, label_visibility='collapsed',
                    help='Click a header above to jump here, or type to filter rows',
                )
                _dp_filt_orig = [c for c in _dp_orig if not _dp_filter_val
                                 or _dp_filter_val.lower() in c.lower()
                                 or _dp_filter_val.lower() in _dp_applied_map.get(c,'').lower()]

                # Build importance lookup from last trained model
                _dp_imp_df = st.session_state.get('_cached_importance_df', pd.DataFrame())
                _dp_imp_lookup: dict = {}
                if not _dp_imp_df.empty and 'Feature' in _dp_imp_df and 'Relative_Importance' in _dp_imp_df:
                    _top_model_imp = _dp_imp_df.groupby('Feature')['Relative_Importance'].max()
                    _imp_max = float(_top_model_imp.max()) if not _top_model_imp.empty else 1.0
                    _dp_imp_lookup = {f: float(v) / max(_imp_max, 1e-9) for f, v in _top_model_imp.items()}
                def _dp_imp_bar(internal_name: str) -> str:
                    ri = _dp_imp_lookup.get(internal_name, 0.0)
                    if ri <= 0: return ''
                    pct = int(round(ri * 100))
                    filled = int(round(ri * 10))
                    bar = '█' * filled + '░' * (10 - filled)
                    return f'{bar} {pct}%'

                # Pre-read editor pending edits to drive the dup banner (only).
                # These are NOT used in the base DataFrame — that stays on applied state.
                _pending = st.session_state.get(_editor_key, {}).get('edited_rows', {})
                for _pidx, _pedits in _pending.items():
                    _prow = int(_pidx) if not isinstance(_pidx, int) else _pidx
                    if 0 <= _prow < len(_dp_filt_orig):
                        _poc = _dp_filt_orig[_prow]
                        if 'Map to' in _pedits:      _dp_eff_map[_poc]  = _pedits['Map to']
                        if 'Column type' in _pedits: _dp_eff_type[_poc] = _pedits['Column type']

                # Dup labels — two separate sets so the base DataFrame (applied) never
                # changes during editing, while the banner still reflects live edits.
                # _dp_eff_dup_labels  → for the warning banner (live pending state)
                # _dp_applied_dup_labels → for the base DataFrame ⚠ column (stable)
                _dp_eff_all_mapped = [
                    v for c, v in _dp_eff_map.items()
                    if v != _DP_KEEP_ORIGINAL and _dp_eff_type.get(c, 'Numeric') != 'Ignore'
                ]
                _dp_eff_dup_labels = {v for v in _dp_eff_all_mapped if _dp_eff_all_mapped.count(v) > 1}
                # ── Auto-Ignore notification ─────────────────────────────
                # When the auto-mapper detected two columns going to the same
                # display label, the second was silently set to Ignore. Show a
                # banner listing what got dropped so the user can override.
                _auto_ig_log = st.session_state.get(_auto_ignored_log_key, [])
                # Only show entries where the column is STILL set to Ignore in
                # the effective state (user may have overridden by now).
                _auto_ig_active = [
                    e for e in _auto_ig_log
                    if _dp_eff_type.get(e['ignored_col']) == 'Ignore'
                ]
                if _auto_ig_active:
                    _ig_lines = [
                        '⚠️ **Auto-Ignored columns** — the auto-mapper detected '
                        'multiple source columns going to the same target and '
                        'silently set the later ones to **Ignore** so processing '
                        'wouldn\'t fail. This may not be what you want — check below:',
                        '',
                    ]
                    for _e in _auto_ig_active:
                        _ig_lines.append(
                            f'• `{_e["ignored_col"]}` was set to **Ignore** because '
                            f'`{_e["kept_col"]}` is also mapped to **{_e["shared_label"]}**.'
                        )
                    _ig_lines += [
                        '',
                        '**Two ways to fix:**',
                        '',
                        '1. **Different target** — if these are genuinely different '
                        'columns (e.g. `FEOwt` → **FeO** vs `FEOTwt` → **FeOt**), '
                        'change the *Map to* dropdown for one of them and Apply.',
                        '',
                        '2. **Keep both** — if both are valid versions of the same '
                        'target (e.g. raw vs anhydrously-recalculated), set the '
                        'Ignored column\'s *Column type* back to **Numeric**, then '
                        'type a unique custom name in its *Rename to* cell '
                        '(e.g. `{0}_anhy`). Apply. The model gets the kept column '
                        'as **{0}**, and you keep the second under your custom '
                        'name for inspection.'.format(
                            _auto_ig_active[0]['shared_label']
                        ),
                    ]
                    st.warning('\n'.join(_ig_lines))

                if _dp_eff_dup_labels:
                    _IRON_DISPLAY_LABELS = {'FeOt', 'FeO', 'Fe2O3', 'Fe2O3t'}
                    _dup_pairs = []
                    _has_iron_dup = False
                    for _dl in sorted(_dp_eff_dup_labels):
                        _cols_with = [c for c, v in _dp_eff_map.items()
                                      if v == _dl and _dp_eff_type.get(c, 'Numeric') != 'Ignore']
                        _dup_pairs.append(f'**{_dl}** ← {", ".join(_cols_with)}')
                        if _dl in _IRON_DISPLAY_LABELS:
                            _has_iron_dup = True
                    _err_lines = [
                        '🚨 **Duplicate mappings — only one source column per target will be used.**',
                        '',
                        'Each line below shows a target with multiple source columns. '
                        'During processing, **only the first source column is kept**; '
                        'the rest are silently dropped, which is almost always not what you want.',
                        '',
                    ] + _dup_pairs + [
                        '',
                        '**To fix:** in the table below, change the *Map to* value for the '
                        'duplicate columns so each target receives exactly one source — or '
                        'set the unwanted column\'s *Column type* to **Ignore**.',
                    ]
                    if _has_iron_dup:
                        _err_lines += [
                            '',
                            '**Iron-specific:** the auto-mapper sometimes labels both '
                            '`FEOwt` (ferrous) and `FEOTwt` (total) as `FeOt`. '
                            'They\'re different — `FEOwt` should map to **FeO**, '
                            '`FEOTwt` to **FeOt**.',
                        ]
                    st.error('\n'.join(_err_lines))

                # Applied-state dup labels — ONLY from committed snapshot.
                # Used in the base DataFrame so it never changes between Apply clicks.
                _dp_applied_all_mapped = [
                    v for c, v in _dp_applied_map.items()
                    if v != _DP_KEEP_ORIGINAL and _dp_applied_type.get(c, 'Numeric') != 'Ignore'
                ]
                _dp_applied_dup_labels = {v for v in _dp_applied_all_mapped if _dp_applied_all_mapped.count(v) > 1}

                # Row-count lookup: stable (from raw upload, never changes).
                _dp_n_total = len(_dp_raw)
                _dp_n_lookup = {c: int(_dp_raw[c].notna().sum()) for c in _dp_raw.columns}

                # Base DataFrame: ALWAYS from applied snapshot — never changes mid-session.
                # Unit column is sourced from the applied unit map (committed) so editing
                # in the table doesn't immediately reprocess the data — the user clicks
                # Apply to commit unit changes alongside mapping changes.
                _dp_unit_options = ['—'] + list(_DP_CONCENTRATION_UNITS)
                def _unit_for_label(_oc, _dl):
                    """Best-effort current Unit cell value for a row.

                    Priority: applied unit map → registry default for the picked
                    label → '—' (no unit applies, e.g. categorical / metadata).
                    """
                    _u = _dp_applied_unit_map.get(_oc) if _dp_applied_unit_map else None
                    if not _u and _dl != _DP_KEEP_ORIGINAL and _dl in _DP_DISPLAY_TO_INTERNAL:
                        _, _u, _ = _DP_DISPLAY_TO_INTERNAL[_dl]
                    return _u if _u in _DP_CONCENTRATION_UNITS else '—'
                _dp_map_df = pd.DataFrame({
                    'Original header': _dp_filt_orig,
                    'Map to':          [_dp_applied_map.get(c, _DP_KEEP_ORIGINAL) for c in _dp_filt_orig],
                    'Unit':             [_unit_for_label(c, _dp_applied_map.get(c, _DP_KEEP_ORIGINAL))
                                         for c in _dp_filt_orig],
                    'Rename to':       [_dp_applied_rename_map.get(c, '') or '' for c in _dp_filt_orig],
                    'Column type':     [_dp_applied_type.get(c, 'Numeric') for c in _dp_filt_orig],
                    'n':               [f'{_dp_n_lookup.get(c, 0):,}/{_dp_n_total:,}' for c in _dp_filt_orig],
                    'Confirmed':       [c in _dp_confirmed or _dp_applied_map.get(c, _DP_KEEP_ORIGINAL) == _DP_KEEP_ORIGINAL for c in _dp_filt_orig],
                    'Model importance':[_dp_imp_bar(_DP_DISPLAY_TO_INTERNAL.get(_dp_applied_map.get(c, ''), (c, None, None))[0]) for c in _dp_filt_orig],
                    '⚠':              ['❌' if (
                                           _dp_applied_map.get(c, _DP_KEEP_ORIGINAL) in _dp_applied_dup_labels
                                           and _dp_applied_type.get(c, 'Numeric') != 'Ignore'
                                           and not (_dp_applied_rename_map.get(c, '') or '').strip()
                                       ) else '' for c in _dp_filt_orig],
                })
                _dp_edited = st.data_editor(
                    _dp_map_df,
                    column_config={
                        'Original header': st.column_config.TextColumn('Original header', disabled=True),
                        'Map to': st.column_config.SelectboxColumn('Map to', options=_DP_DISPLAY_LABELS, required=True),
                        'Unit': st.column_config.SelectboxColumn(
                            'Unit',
                            options=_dp_unit_options,
                            required=True,
                            help=(
                                'Concentration unit of the *original* values. Auto-detected '
                                'from the column header (`(wt%)`, `(ppm)`, `(ppb)` etc.) '
                                'and used to convert into the canonical unit when Apply runs:\n\n'
                                '• Major oxides → `wt%` (1 wt% = 10,000 ppm)\n\n'
                                '• Trace elements → `ppm` (1 ppm = 1000 ppb)\n\n'
                                'Pick `—` for non-concentration columns (categorical, '
                                'metadata, location, age). Override if auto-detection got it '
                                'wrong (e.g. file actually reports `Sr` in ppb but the header '
                                'says ppm).'
                            ),
                        ),
                        'Rename to': st.column_config.TextColumn(
                            'Rename to',
                            help=(
                                'Optional custom output column name. Leave blank to use the '
                                '**Map to** value as the output name (the standard path — what '
                                'every formula expects).\n\n'
                                'Use this when you have **two source columns mapping to the '
                                'same target** and want to keep both. Example: your file has '
                                'both raw `SIO2wt` and pre-recalculated `SiO2 (wt%)`. Map both '
                                'to `SiO2`, then type `SiO2_anhy` here for the second one — '
                                'now the model gets clean `SiO2`, and you keep the '
                                'pre-recalculated values under `SiO2_anhy` for inspection.\n\n'
                                'Conversion (unit + element→oxide) still applies based on '
                                '*Map to* + *Unit*; only the **final output column name** '
                                'changes.'
                            ),
                        ),
                        'Column type': st.column_config.SelectboxColumn(
                            'Column type',
                            options=['Numeric', 'Category', 'Metadata', 'Ignore'],
                            required=True,
                            help=(
                                'Numeric — measurements and quantities (elements, age, coordinates). '
                                'Available for ML, GAME, plots, maps, and range/bin grouping.\n\n'
                                'Category — discrete labels (tectonic setting, arc, rock type). '
                                'Available for colouring and categorical grouping.\n\n'
                                'Metadata — identifiers and notes unique to each row (sample ID, reference). '
                                'Passes through but excluded from analysis and grouping.\n\n'
                                'Ignore — column is completely dropped from the output. '
                                'Use this to remove duplicates or unwanted columns.'
                            ),
                        ),
                        'n': st.column_config.TextColumn('n', disabled=True,
                            help='Non-null rows / total rows in this column'),
                        'Confirmed': st.column_config.CheckboxColumn('Confirmed'),
                        'Model importance': st.column_config.TextColumn('Model importance', disabled=True,
                            help='Relative importance of this feature in the last trained model (blank = not a model feature or no model trained yet)'),
                        '⚠': st.column_config.TextColumn('⚠', disabled=True,
                            help='Duplicate: same target mapped to more than one active column'),
                    },
                    hide_index=True, use_container_width=True,
                    key=_editor_key,
                )

                # Compute pending-change counts (for the Apply button badge).
                # Compare editor return value against the applied snapshot.
                if not _dp_edited.empty:
                    _e_map  = dict(zip(_dp_edited['Original header'], _dp_edited['Map to']))
                    _e_type = dict(zip(_dp_edited['Original header'], _dp_edited['Column type']))
                    _dp_n_pending_map  = sum(1 for c in _dp_filt_orig if _e_map.get(c)  != _dp_applied_map.get(c))
                    _dp_n_pending_type = sum(1 for c in _dp_filt_orig if _e_type.get(c) != _dp_applied_type.get(c))
                    _dp_has_pending = bool(_dp_n_pending_map or _dp_n_pending_type)

                # ── Export / Edit-in-Excel ───────────────────────────────────
                _exp_c1, _exp_c2 = st.columns(2)
                _tpl_bytes = json.dumps(_dp_applied_map, indent=2, ensure_ascii=False).encode('utf-8')
                _exp_c1.download_button(
                    'Export mapping template (.json)',
                    data=_tpl_bytes,
                    file_name=f'mapping_template_{_dp_f.name}.json',
                    mime='application/json',
                    key=f'{_dp_key}_tpl_dl_{_dp_fi}_{_dp_f.name}',
                    help='Save mapping as a JSON template to re-use with similar files',
                )
                _xl_bytes = _dp_mapping_to_excel(_dp_orig, _dp_applied_map, _dp_confirmed)
                if _xl_bytes is not None:
                    _exp_c2.download_button(
                        'Edit in Excel (.xlsx)',
                        data=_xl_bytes,
                        file_name=f'mapping_{_dp_f.name}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key=f'{_dp_key}_xl_dl_{_dp_fi}_{_dp_f.name}',
                        help='Download mapping as Excel with validated dropdowns — edit and re-import via "Import mapping template"',
                    )
                else:
                    _exp_c2.caption('Install openpyxl to enable Edit in Excel')

            # UTM conversion
            _dp_utm_info = None
            if 'UTM Easting [m]' in _dp_applied_map.values() and 'UTM Northing [m]' in _dp_applied_map.values():
                st.info('UTM coordinates detected — specify zone to convert to WGS84 DD.')
                _uc1, _uc2, _uc3 = st.columns([1,2,2])
                _dp_utm_zone = _uc1.number_input('UTM Zone', 1, 60, 30, 1, key=f'{_dp_key}_utm_zone_{_dp_fi}_{_dp_f.name}')
                _dp_utm_hemi = _uc2.radio('Hemisphere', ['Northern','Southern'], horizontal=True, key=f'{_dp_key}_utm_hemi_{_dp_fi}_{_dp_f.name}')
                _dp_utm_info = (int(_dp_utm_zone), _dp_utm_hemi == 'Northern')
                _uc3.caption(f'WGS84 {int(_dp_utm_zone)}{"N" if _dp_utm_info[1] else "S"} → DD')

            # ── Apply / Undo / Confirm all / Reset ───────────────────────────
            _ba0, _ba1, _ba2, _ba3 = st.columns(4)
            if _ba0.button(
                '▶ Apply',
                key=f'{_dp_key}_apply_{_dp_fi}_{_dp_f.name}',
                type='primary' if _dp_has_pending else 'secondary',
                help='Apply current mapping, units, and column types to the data. '
                     'Changes in the table are only used downstream once you click this.',
            ):
                # Build new applied state from the editor return value
                _apply_map    = dict(_dp_applied_map)
                _apply_type   = dict(_dp_applied_type)
                _apply_unit   = dict(_dp_applied_unit_map)
                _apply_rename = dict(_dp_applied_rename_map)
                _apply_conf   = set(_dp_confirmed)
                if not _dp_edited.empty:
                    _e_map    = dict(zip(_dp_edited['Original header'], _dp_edited['Map to']))
                    _e_type   = dict(zip(_dp_edited['Original header'], _dp_edited['Column type']))
                    _e_unit   = (dict(zip(_dp_edited['Original header'], _dp_edited['Unit']))
                                 if 'Unit' in _dp_edited.columns else {})
                    _e_rename = (dict(zip(_dp_edited['Original header'], _dp_edited['Rename to']))
                                 if 'Rename to' in _dp_edited.columns else {})
                    _e_conf   = set(_dp_edited.loc[_dp_edited['Confirmed'], 'Original header'])
                    _filt_set = set(_dp_filt_orig)
                    for _oc in _dp_filt_orig:
                        _new_dl = _e_map.get(_oc, _dp_applied_map.get(_oc, _DP_KEEP_ORIGINAL))
                        _old_dl = _dp_applied_map.get(_oc, _DP_KEEP_ORIGINAL)
                        # Save user alias when mapping changes
                        if _new_dl != _old_dl:
                            _dp_save_user_alias(_oc, _new_dl)
                            # Auto-derive type for changed mapping
                            if _new_dl != _DP_KEEP_ORIGINAL and _new_dl in _DP_DISPLAY_TO_REGCAT:
                                _apply_type[_oc] = _DP_REG_CAT_TO_TYPE.get(_DP_DISPLAY_TO_REGCAT[_new_dl], 'Numeric')
                            else:
                                _apply_type[_oc] = _dp_auto_type([_oc], {_oc: _DP_KEEP_ORIGINAL}, _dp_raw).get(_oc, 'Numeric')
                        else:
                            # Mapping unchanged — use explicit type from editor
                            _apply_type[_oc] = _e_type.get(_oc, _dp_applied_type.get(_oc, 'Numeric'))
                        _apply_map[_oc] = _new_dl
                        # Commit the user-picked unit. '—' means "no unit applies"
                        # (e.g. categorical/metadata) and is stored as None so the
                        # apply-mapping code falls back to the registry default.
                        _u_picked = _e_unit.get(_oc)
                        _apply_unit[_oc] = (_u_picked if _u_picked in _DP_CONCENTRATION_UNITS else None)
                        # Commit the user-typed Rename-to. Empty string is
                        # equivalent to "no rename" — keys stripped of whitespace.
                        _r_picked = _e_rename.get(_oc, '')
                        _r_clean  = _r_picked.strip() if isinstance(_r_picked, str) else ''
                        if _r_clean:
                            _apply_rename[_oc] = _r_clean
                        else:
                            _apply_rename.pop(_oc, None)
                    _apply_conf = {c for c in _dp_confirmed if c not in _filt_set} | _e_conf
                st.session_state[_dp_applied_mkey]      = _apply_map
                st.session_state[_dp_applied_tkey]      = _apply_type
                st.session_state[_dp_applied_unit_key]  = _apply_unit
                st.session_state[_dp_applied_rename_key] = _apply_rename
                st.session_state[_dp_mkey]  = _apply_map   # keep in sync for template export etc.
                st.session_state[_dp_tkey]  = _apply_type
                st.session_state[_dp_ukey]  = _apply_unit
                st.session_state[_dp_rkey]  = _apply_rename
                st.session_state[_dp_ckey]  = _apply_conf
                st.rerun()
            if _ba1.button(
                '↩ Undo',
                key=f'{_dp_key}_undo_{_dp_fi}_{_dp_f.name}',
                disabled=not _dp_has_pending,
                help='Discard unapplied changes and restore the last applied state.',
            ):
                # Clearing the editor widget's session-state key resets its
                # edited_rows to {} so it reverts to the base (applied) DataFrame.
                st.session_state.pop(_editor_key, None)
                st.rerun()
            if _ba2.button('Confirm all', key=f'{_dp_key}_confirm_all_{_dp_fi}_{_dp_f.name}',
                           help='Mark all columns as reviewed (does not apply mapping)'):
                st.session_state[_dp_ckey] = set(_dp_orig)
                st.session_state[_dp_card_key] = False
                st.rerun()
            if _ba3.button('Reset mapping', key=f'{_dp_key}_reset_{_dp_fi}_{_dp_f.name}'):
                for _rk in [_dp_mkey, _dp_ckey, _dp_tkey, _dp_ukey, _dp_rkey,
                            _dp_applied_mkey, _dp_applied_tkey,
                            _dp_applied_unit_key, _dp_applied_rename_key,
                            _auto_ignored_log_key]:
                    st.session_state.pop(_rk, None)
                st.rerun()
            if _dp_has_pending:
                _chg_parts: list = []
                if _dp_n_pending_map:  _chg_parts.append(f'{_dp_n_pending_map} mapping{"s" if _dp_n_pending_map  != 1 else ""}')
                if _dp_n_pending_type: _chg_parts.append(f'{_dp_n_pending_type} type{"s"  if _dp_n_pending_type != 1 else ""}')
                st.info(f'📝 {" and ".join(_chg_parts) or "Changes"} pending — click **▶ Apply** to update the data.')

            # Tab assignment
            _dp_use_key = f'{_dp_key}_use_{_dp_fi}_{_dp_f.name}'
            if _dp_use_key not in st.session_state:
                st.session_state[_dp_use_key] = ['Model (training)', 'Validate', 'Predict']
            _dp_use = st.multiselect(
                'Use in tab(s)',
                ['Model (training)', 'Validate', 'Predict'],
                default=st.session_state[_dp_use_key],
                key=_dp_use_key,
                help='Assign this file to one or more workflow tabs — uncheck any you do not want this file routed to',
            )

            # ── Processed DataFrame — cached by APPLIED mapping hash only ──────
            # Keyed on the applied (committed) map so that editing in the table
            # does not trigger an expensive recalculation until Apply is clicked.
            # Unit map is included in the cache signature so changing a Unit
            # cell + Apply forces recomputation.
            import hashlib as _hl
            _dp_ignore_proc_cols = {
                _oc for _oc, _tp in _dp_applied_type.items() if _tp == 'Ignore'
            }
            _dp_proc_key = f'{_dp_key}_proc_{_dp_fi}_{_dp_f.name}'
            _dp_map_hash    = _hl.md5(json.dumps(sorted(_dp_applied_map.items()),  ensure_ascii=False).encode()).hexdigest()
            _dp_type_hash   = _hl.md5(json.dumps(sorted(_dp_applied_type.items()), ensure_ascii=False).encode()).hexdigest()
            _dp_unit_hash   = _hl.md5(json.dumps(sorted(_dp_applied_unit_map.items()), ensure_ascii=False).encode()).hexdigest()
            _dp_rename_hash = _hl.md5(json.dumps(sorted(_dp_applied_rename_map.items()), ensure_ascii=False).encode()).hexdigest()
            _dp_utm_hash    = str(_dp_utm_info)
            _dp_proc_sig    = f'{_dp_map_hash}_{_dp_type_hash}_{_dp_unit_hash}_{_dp_rename_hash}_{_dp_utm_hash}'
            _dp_proc_sig_key = f'{_dp_proc_key}_sig'
            if (st.session_state.get(_dp_proc_sig_key) != _dp_proc_sig or
                    _dp_proc_key not in st.session_state):
                try:
                    st.session_state[_dp_proc_key] = _dp_apply_mapping(
                        _dp_raw, _dp_applied_map, _dp_utm_info,
                        unit_map=_dp_applied_unit_map,
                        ignore_cols=_dp_ignore_proc_cols,
                        rename_map=_dp_applied_rename_map,
                    )
                    st.session_state[_dp_proc_sig_key] = _dp_proc_sig
                except Exception as _dp_map_err:
                    st.error(f'Error processing mapping: {_dp_map_err}')
                    st.session_state[_dp_proc_key] = _dp_raw.copy()
                    st.session_state[_dp_proc_sig_key] = _dp_proc_sig
            _dp_processed = st.session_state[_dp_proc_key]

            # ── Data Quality panel ───────────────────────────────────────────
            with st.expander('Data quality check', expanded=False):
                _dp_qa_panel(_dp_processed, key_prefix=f'{_dp_key}_{_dp_fi}_{_dp_f.name}')

            # ── Anhydrous toggle (prominent — default ON) ────────────────────
            _anhy_key_now = f'{_dp_key}_{_dp_fi}_{_dp_f.name}_anhy'
            _has_major_now = any(c in _dp_processed.columns for c in _DP_MAJOR_OXIDE_ANHY)
            if _has_major_now and _anhy_key_now not in st.session_state:
                st.session_state[_anhy_key_now] = True   # on by default
            if _has_major_now:
                st.checkbox(
                    '🔁 Normalise major oxides to 100 % anhydrous (recommended)',
                    key=_anhy_key_now,
                    help=(
                        'Each oxide is recalculated as oxide / Σ(anhydrous oxides) × 100 '
                        '(LOI and H₂Ot excluded from the sum), rounded to 2 d.p. '
                        'This is the standard preparation for thermodynamic modelling '
                        'and petrogenetic discrimination. Uncheck only if you need raw '
                        'analytical values.'
                    ),
                )

            # ── Major element QC panel ────────────────────────────────────────
            with st.expander('Major element QC', expanded=False):
                _dp_major_qc_panel(_dp_processed, key_prefix=f'{_dp_key}_{_dp_fi}_{_dp_f.name}')

            # ── Geologic age consistency check ────────────────────────────────
            # Flag rows where numeric Age_Ma disagrees with a named era/period/
            # epoch column (e.g. Age_Ma=120 but Geologic_Period='Triassic').
            _geo_chk_cols = [c for c in ['Geologic_Era', 'Geologic_Period', 'Geologic_Epoch']
                             if c in _dp_processed.columns]
            if 'Age_Ma' in _dp_processed.columns and _geo_chk_cols:
                # Build Ma-range lookups from GEO_TIME_BINS (epoch/period/era)
                _geo_era_rng:    dict = {}
                _geo_period_rng: dict = {}
                _geo_epoch_rng:  dict = {}
                for _gep, _gper, _gera, _gyo, _gol in GEO_TIME_BINS:
                    _geo_epoch_rng[_gep.lower()] = (_gyo, _gol)
                    _k = _gper.lower()
                    _geo_period_rng[_k] = (
                        min(_geo_period_rng.get(_k, (_gyo, _gol))[0], _gyo),
                        max(_geo_period_rng.get(_k, (_gyo, _gol))[1], _gol),
                    )
                    _k = _gera.lower()
                    _geo_era_rng[_k] = (
                        min(_geo_era_rng.get(_k, (_gyo, _gol))[0], _gyo),
                        max(_geo_era_rng.get(_k, (_gyo, _gol))[1], _gol),
                    )
                _geo_col_rng = {
                    'Geologic_Era':    _geo_era_rng,
                    'Geologic_Period': _geo_period_rng,
                    'Geologic_Epoch':  _geo_epoch_rng,
                }
                # Vectorised conflict detection (per-column .map() + boolean
                # mask). Replaces a Python row-by-row loop that ran N × 3
                # iterations on every Prepare rerun.
                _geo_age = pd.to_numeric(_dp_processed['Age_Ma'], errors='coerce')
                _geo_pos = np.arange(len(_dp_processed))
                _geo_sid = (_dp_processed['Sample_ID'].astype(str)
                            if 'Sample_ID' in _dp_processed.columns
                            else pd.Series(_geo_pos.astype(str), index=_dp_processed.index))
                _geo_conflict_frames = []
                for _gc in _geo_chk_cols:
                    _rng_lkp = _geo_col_rng[_gc]
                    _key_ser = _dp_processed[_gc].astype(str).str.strip().str.lower()
                    _lo_ser  = _key_ser.map(lambda k, _r=_rng_lkp: _r[k][0] if k in _r else np.nan)
                    _hi_ser  = _key_ser.map(lambda k, _r=_rng_lkp: _r[k][1] if k in _r else np.nan)
                    _mask = (
                        _geo_age.notna()
                        & _dp_processed[_gc].notna()
                        & _lo_ser.notna()
                        & ((_geo_age < _lo_ser) | (_geo_age >= _hi_ser))
                    ).to_numpy()
                    if _mask.any():
                        _geo_conflict_frames.append(pd.DataFrame({
                            'Row':               _geo_pos[_mask],
                            'Sample_ID':         _geo_sid.to_numpy()[_mask],
                            'Age_Ma':            _geo_age.to_numpy()[_mask],
                            _gc:                 _dp_processed[_gc].astype(str).to_numpy()[_mask],
                            'Expected Ma range': [f'{lo}–{hi}' for lo, hi in
                                                  zip(_lo_ser.to_numpy()[_mask],
                                                      _hi_ser.to_numpy()[_mask])],
                        }))
                if _geo_conflict_frames:
                    _geo_conflict_df = pd.concat(_geo_conflict_frames, ignore_index=True)
                    with st.expander(
                        f'⚠️ Geologic age conflicts — {len(_geo_conflict_df)} row(s)',
                        expanded=False,
                    ):
                        st.caption(
                            '`Age_Ma` falls outside the expected Ma range for the named '
                            'geologic label. Common causes: Ka entered as Ma (divide by 1000), '
                            'typo in era/period name, or mismatched column assignment.'
                        )
                        st.dataframe(
                            _geo_conflict_df,
                            hide_index=True, use_container_width=True,
                        )

            # ── Build export df (anhydrous + row-ignore applied) ─────────────
            # Column-ignore is now handled inside _dp_apply_mapping via the
            # ``ignore_cols`` argument — see processed-DataFrame block above.
            # That fix prevents a duplicate-mapped sibling (e.g. ``SIO2wt``
            # next to an Ignored ``SiO2 (wt%)``) from getting collateral-
            # damaged when the post-processing tried to drop ``SiO2`` by
            # internal name.
            _anhy_on_now = st.session_state.get(_anhy_key_now, _has_major_now)
            _dp_display  = _dp_anhydrous_recalc(_dp_processed) if _anhy_on_now else _dp_processed
            _dp_ignored  = st.session_state.get(f'{_dp_key}_{_dp_fi}_{_dp_f.name}_ignore', set())
            _dp_export_df = _dp_display.loc[~_dp_display.index.isin(_dp_ignored)].reset_index(drop=True)

            # ── Column-completeness filter ────────────────────────────────────
            # Imports like Luffi T1 ship 190+ columns, many of which are
            # >90 % blank for a given sample subset (e.g. He isotopes only
            # measured on a handful of arc samples). Surface a threshold
            # picker so the user can drop the noise in one click rather
            # than ignoring columns one-by-one in the mapping editor.
            #
            # Threshold semantics: "drop columns where more than X% of
            # the values are missing". 100 % keeps everything (default).
            # 0 % drops any column with even one missing value.
            _missing_options  = ['Keep all'] + [f'> {p} %' for p in range(10, 101, 10)] + ['Strict (any missing)']
            _missing_key = f'{_dp_key}_{_dp_fi}_{_dp_f.name}_missing_thresh'
            _missing_choice = st.selectbox(
                '🧹 Column completeness filter',
                _missing_options,
                index=0,
                key=_missing_key,
                help='Drops columns whose missing-value percentage **exceeds** '
                     'the chosen threshold. "> 90 %" keeps everything except '
                     'near-blank columns; "> 10 %" keeps only the most complete '
                     'columns. Useful for cleaning up big imports like Luffi T1 '
                     'where most isotope/PGE columns are sparsely populated.',
            )
            if _missing_choice == 'Keep all':
                _missing_thresh_pct = None
            elif _missing_choice == 'Strict (any missing)':
                _missing_thresh_pct = 0.0
            else:
                # Parse "> N %" → N as float
                try:
                    _missing_thresh_pct = float(_missing_choice.split('>')[1].split('%')[0].strip())
                except Exception:
                    _missing_thresh_pct = None
            if _missing_thresh_pct is not None and len(_dp_export_df) > 0:
                _missing_pct_per_col = _dp_export_df.isna().mean() * 100.0
                _drop_mask = _missing_pct_per_col > _missing_thresh_pct
                _dropped_cols = _missing_pct_per_col[_drop_mask].sort_values(ascending=False)
                if len(_dropped_cols) > 0:
                    _dp_export_df = _dp_export_df.drop(columns=list(_dropped_cols.index))
                    # Brief, informative summary — first 6 worst offenders by
                    # name + their missingness, plus the count overflow.
                    _preview = ', '.join(
                        f'`{c}` ({pct:.0f}%)' for c, pct in _dropped_cols.head(6).items()
                    )
                    if len(_dropped_cols) > 6:
                        _preview += f', +{len(_dropped_cols) - 6} more'
                    st.caption(
                        f'Dropped **{len(_dropped_cols)}** column(s) with > '
                        f'{_missing_thresh_pct:.0f}% missing values: {_preview}'
                    )
                else:
                    st.caption(f'No columns exceed the {_missing_thresh_pct:.0f}% missingness threshold.')

            # ── Processed data download ──────────────────────────────────────
            st.download_button(
                f'Download processed data ({len(_dp_export_df):,} rows)',
                _dp_export_df.rename(columns=_EXPORT_COL_LABELS).to_csv(index=False).encode('utf-8'),
                file_name=f'processed_{_dp_f.name.rsplit(".",1)[0]}.csv',
                mime='text/csv',
                key=f'{_dp_key}_proc_dl_{_dp_fi}_{_dp_f.name}',
                help='Mapped, unit-converted, ratio-computed data ready for review',
            )

        if 'Model (training)' in st.session_state.get(_dp_use_key, []): _dp_dest_dfs['dp_training_df'].append(_dp_export_df)
        if 'Validate'         in st.session_state.get(_dp_use_key, []): _dp_dest_dfs['dp_validation_df'].append(_dp_export_df)
        if 'Predict'          in st.session_state.get(_dp_use_key, []): _dp_dest_dfs['dp_prediction_df'].append(_dp_export_df)
        # Expose each processed file as its own named pool entry (stem = name without extension).
        # This lets users pick individual sheets in the Model / Validate / Predict source dropdowns.
        _dp_stem = _dp_f.name.rsplit('.', 1)[0]
        _dp_individual_sheets[_dp_stem] = _dp_export_df
        _dp_individual_roles[_dp_stem] = set(st.session_state.get(_dp_use_key, ['Model (training)', 'Validate', 'Predict']))

    # ── Build merged column-type registry across all uploaded files ───────────
    # Maps processed column name → 'Numeric' | 'Category' | 'Metadata'
    # Used by Grouping and other tabs to respect user's type assignments.
    _dp_merged_types: dict = {}
    for _pfi, _pf in enumerate(_dp_uploads or []):
        _pmkey = f'{_dp_key}_map_{_pfi}_{_pf.name}'
        _ptkey = f'{_dp_key}_type_{_pfi}_{_pf.name}'
        _pm  = st.session_state.get(_pmkey, {})
        _pt  = st.session_state.get(_ptkey, {})
        for _oc, _dl in _pm.items():
            _col_type = _pt.get(_oc, 'Numeric')
            if _col_type == 'Ignore':
                continue  # column has been dropped — don't register it
            _col_name = _DP_DISPLAY_TO_INTERNAL[_dl][0] if _dl != _DP_KEEP_ORIGINAL and _dl in _DP_DISPLAY_TO_INTERNAL else _oc
            _dp_merged_types[_col_name] = _col_type
    st.session_state['dp_column_types'] = _dp_merged_types

    # Merge and store per-destination DataFrames
    _dp_summary_parts = []
    for _dest_key, _dest_list in _dp_dest_dfs.items():
        _dest_label = {'dp_training_df':'Model','dp_validation_df':'Validate','dp_prediction_df':'Predict'}[_dest_key]
        if _dest_list:
            try:
                _merged = pd.concat(_dest_list, ignore_index=True) if len(_dest_list) > 1 else _dest_list[0]
                st.session_state[_dest_key] = _merged
                _dp_summary_parts.append(f'**{_dest_label}** {len(_merged):,} rows')
            except Exception as _e:
                st.warning(f'Could not merge files for {_dest_label}: {_e}')
        else:
            # No files routed to this destination — clear any stale value
            st.session_state.pop(_dest_key, None)

    # ── Prepped-datasets pool ────────────────────────────────────────────────
    # Single shared dict that Group / Model / Validate / Predict tabs read from.
    # Each uploaded file becomes ONE entry, keyed by its filename stem — no
    # role-aggregated "Training data"/"Validation data"/"Prediction data"
    # buckets. Role assignment (which tabs see this file) lives separately in
    # st.session_state['_dp_dataset_roles']. Any non-Prepare entries (e.g.
    # grouped derivatives from the Group tab) survive untouched so the user
    # doesn't lose them when a new file is added in Prepare.
    _pool = dict(st.session_state.get('prepped_datasets', {}))
    # Defensive: clear any stale legacy bucket entries left over from previous
    # sessions or older app versions.
    for _stale in _PREPARE_POOL_KEYS:
        _pool.pop(_stale, None)
    # Snapshot any Group_ID / Group_Name columns the Group tab has applied to
    # the existing pool entries BEFORE we wipe-and-rebuild them. Without this,
    # every Streamlit rerun (which re-executes the Prepare tab block) would
    # overwrite the user's grouped pool entry with freshly-processed data and
    # silently drop the Group_ID column — making the Model tab's
    # `_pool_has_groups` check return False and hiding multi-model training.
    _grouping_snapshots: dict = {}
    for _stale_sheet in st.session_state.get('_dp_sheet_pool_keys', []):
        _existing = _pool.get(_stale_sheet)
        if isinstance(_existing, pd.DataFrame):
            _snap = {}
            for _gcol in ('Group_ID', 'Group_Name'):
                if _gcol in _existing.columns:
                    _snap[_gcol] = _existing[_gcol].copy()
            if _snap:
                _grouping_snapshots[_stale_sheet] = (_snap, len(_existing))
        _pool.pop(_stale_sheet, None)
    # Individual processed sheets — re-add fresh, merging back any Group_ID /
    # Group_Name we just snapshotted (only when row counts match — so a true
    # re-upload of a different file size cleanly drops the stale grouping).
    _new_sheet_pool_keys = []
    for _sheet_stem, _sheet_df in _dp_individual_sheets.items():
        _snap_pair = _grouping_snapshots.get(_sheet_stem)
        if _snap_pair is not None:
            _snap, _snap_len = _snap_pair
            if _snap_len == len(_sheet_df):
                _sheet_df = _sheet_df.copy()
                for _gcol, _gvals in _snap.items():
                    if _gcol not in _sheet_df.columns:
                        _sheet_df[_gcol] = _gvals.values
        _pool[_sheet_stem] = _sheet_df
        _new_sheet_pool_keys.append(_sheet_stem)
    st.session_state['_dp_sheet_pool_keys'] = _new_sheet_pool_keys
    st.session_state['prepped_datasets'] = _pool
    # Persist role assignments so Model / Validate / Predict tabs can filter
    st.session_state['_dp_dataset_roles'] = _dp_individual_roles

    if _dp_uploads and _dp_summary_parts:
        st.success('Ready — ' + ' · '.join(_dp_summary_parts))
    elif _dp_uploads:
        st.info('Files loaded but none assigned to a tab yet. Use the "Use in tab(s)" selector above.')

    # ── Column provenance ─────────────────────────────────────────────────────
    with st.expander('Column provenance', expanded=False):
        st.caption('Every transformation applied to uploaded columns during mapping.')
        _prov_rows = []
        for _pfi, _pf in enumerate(st.session_state.get('dp_global_uploader') or []):
            _pmkey = f'dp_map_{_pfi}_{_pf.name}'
            _pm = st.session_state.get(_pmkey, {})
            for _oc, _dl in _pm.items():
                if _dl == _DP_KEEP_ORIGINAL:
                    _prov_rows.append({'File': _pf.name, 'Original column': _oc, 'Mapped to': _oc, 'Transformation': 'none (kept as-is)'})
                elif _dl in _DP_DISPLAY_TO_INTERNAL:
                    _in, _du, _cv = _DP_DISPLAY_TO_INTERNAL[_dl]
                    if _cv is None:
                        _tx = f'rename ({_du})' if _du else 'rename'
                    elif _cv[0] == 'elem_oxide':
                        _tx = f'element → oxide (× {_cv[1]:.4f}, {_du})'
                    elif _cv[0] == 'utm':
                        _tx = 'UTM → WGS84 DD'
                    else:
                        _tx = str(_cv)
                    _prov_rows.append({'File': _pf.name, 'Original column': _oc, 'Mapped to': _in, 'Display label': _dl, 'Default unit': _du or '—', 'Transformation': _tx})
        if _prov_rows:
            _prov_df = pd.DataFrame(_prov_rows)
            st.dataframe(_prov_df, hide_index=True, use_container_width=True)
            st.download_button('Download provenance CSV', _prov_df.to_csv(index=False).encode('utf-8'),
                               'column_provenance.csv', 'text/csv', key='dl_provenance')
        else:
            st.info('No mappings confirmed yet.')

with t0:
    # ── 1. Training dataset ───────────────────────────────────────────────────
    # Built-in reference calibration sets are always available.
    # User datasets from the Prepare tab appear here only when assigned
    # "Model (training)" in Prepare.  No file uploader — upload via Prepare tab.
    source_options = ['Guo & Yang (2023)', 'Zou et al. (2021)']
    if find_luffi_training():
        source_options.append('Luffi & Ducea (2022)')
    # Hide pool entries whose stem matches a built-in — selecting the
    # built-in's friendly name will pick up the prepared version automatically
    # via read_training_source's pool-preference path. Showing both would be
    # a duplicate UX.
    _builtin_stems_in_dropdown = {
        _BUILTIN_TO_FILE_STEM[_b] for _b in source_options if _b in _BUILTIN_TO_FILE_STEM
    }
    for _pn in get_pool_for_tab('Model (training)'):
        if _pn in source_options or _pn in _builtin_stems_in_dropdown:
            continue
        source_options.append(_pn)
    _src_c, _rel_c = st.columns([0.82, 0.18])
    primary_source = _src_c.selectbox(
        '1. Training dataset', source_options, index=0,
        key='primary_training_source',
        help='Calibration dataset with known crustal thicknesses. '
             'Upload your own data in the **Prepare** tab and assign it "Model (training)".',
    )
    if _rel_c.button('↻ Reload', key='reload_training_data', use_container_width=True,
                     help='Clear the cached file read and reload from disk.'):
        read_table.clear()
        enrich.clear()
    primary_upload = None   # all uploads come through the Prepare tab
    if primary_source in get_prepped_pool():
        _ps_df = get_prepped_pool()[primary_source]
        st.info(f"Using **{primary_source}** from Prepare tab: {len(_ps_df):,} rows.")

    # ── 2. Training approach ──────────────────────────────────────────────────
    _pool_has_groups = any(
        'Group_ID' in _pdf.columns
        for _pdf in get_prepped_pool().values()
    )
    _ml_approach = (
        st.radio(
            '2. Training approach',
            ['Single model', 'Multi-model — train per group'],
            horizontal=True,
            key='ml_train_approach',
            help=(
                '**Single model** — one model trained on the whole dataset '
                '(or a chosen subset). Best for general-purpose prediction.\n\n'
                '**Multi-model** — a separate ML model for each group or '
                'combination of groups. Load a grouped dataset from the '
                '**Group** tab to unlock this option.'
            ),
        )
        if _pool_has_groups
        else 'Single model'
    )
    if not _pool_has_groups:
        st.caption(
            '💡 Use the **Group** tab to define groups and save to pool — '
            'Multi-model training will then be available here.'
        )
    elif _ml_approach == 'Multi-model — train per group':
        st.caption(
            '📍 **Per-group model assignment** panel is at the bottom of this tab — '
            'configure which trained model applies to each group after training completes.'
        )

    # ── 3. Model type (single-model only; multi-model is always ML) ──────────
    _pmt_is_ml = True          # default — multi-model path always uses ML
    _pmt = 'Machine Learning'  # default — keeps ratio/map branches dormant
    if _ml_approach == 'Single model':
        _pmt_opts = ['Machine Learning', 'Multi-ratio', 'Single-ratio']
        _pmt = st.radio(
            '3. Model type', _pmt_opts, index=0, horizontal=True,
            key='primary_model_type',
            help=(
                '**Machine Learning** — train an ensemble ML model '
                '(ExtraTrees, RandomForest, XGBoost…) on a geochemical calibration dataset.\n\n'
                '**Multi-ratio** — GAME consensus (Luffi & Ducea 2022) or Sundell et al. (2021) '
                'paired Sr/Y + La/Yb(N) — two element ratios combined.\n\n'
                '**Single-ratio** — Profeta, Sundell, Zou (Sr/Y or La/Yb), '
                'or Mantle & Collins (Ce/Y) — single element ratio.'
            ),
        )
        _pmt_is_ml = (_pmt == 'Machine Learning')

    # ── Ratio / GAME panel (non-ML, single-model mode only) ─────────────────
    if _ml_approach == 'Single model' and not _pmt_is_ml:
        _mr_opts = {
            'GAME consensus — Luffi & Ducea (2022)': 'H_GAME_LuffiDucea2022_km',
            'Sundell et al. (2021) paired Sr/Y + La/Yb(N)': 'H_Sundell2021_Paired_km',
        }
        _sr_opts = {
            'Profeta et al. (2015) — Sr/Y': 'H_Profeta2015_SrY_km',
            'Sundell et al. (2021) — Sr/Y': 'H_Sundell2021_SrY_km',
            'Zou et al. (2021) — Sr/Y': 'H_Zou2021_SrY_SVRE_km',
            'Profeta et al. (2015) — La/Yb(N)': 'H_Profeta2015_LaYbN_km',
            'Sundell et al. (2021) — La/Yb(N)': 'H_Sundell2021_LaYbN_km',
            'Zou et al. (2021) — La/Yb(N)': 'H_Zou2021_LaYbN_SVRE_km',
            'Mantle & Collins (2008) — Ce/Y': 'H_Mantle2008_CeY_sample_km',
        }
        _ratio_opts = _mr_opts if _pmt == 'Multi-ratio' else _sr_opts
        _ratio_key = 'primary_mr_model' if _pmt == 'Multi-ratio' else 'primary_sr_model'
        _ratio_choice = st.selectbox(
            f'{"Multi-ratio" if _pmt == "Multi-ratio" else "Single-ratio"} model',
            list(_ratio_opts.keys()), key=_ratio_key,
        )
        _primary_proxy_col = _ratio_opts[_ratio_choice]
        st.session_state['_primary_proxy_col'] = _primary_proxy_col
        st.session_state['_primary_proxy_name'] = _ratio_choice
        _ratio_info = {
            'H_GAME_LuffiDucea2022_km':   ('La, Ce, Nd, Yb, Sr, Y… (up to 28 GAME sensors)', 'GAME consensus across multiple mohometers'),
            'H_Sundell2021_Paired_km':    ('Sr, Y, La, Yb', 'H = 10.3 ln(Sr/Y) + 8.8 ln(La/Yb(N)) − 10.6'),
            'H_Profeta2015_SrY_km':       ('Sr, Y', 'H = (Sr/Y + 7.25) / 0.90'),
            'H_Sundell2021_SrY_km':       ('Sr, Y', 'H = 19.6 ln(Sr/Y) − 24'),
            'H_Zou2021_SrY_SVRE_km':      ('Sr, Y', 'H = 1.11(Sr/Y) + 8.05'),
            'H_Profeta2015_LaYbN_km':     ('La, Yb', 'H = ln(La/Yb(N) / 0.98) / 0.047'),
            'H_Sundell2021_LaYbN_km':     ('La, Yb', 'H = 17 ln(La/Yb(N)) + 6.9'),
            'H_Zou2021_LaYbN_SVRE_km':    ('La, Yb', 'H = 21.277 ln(1.0204 La/Yb(N))'),
            'H_Mantle2008_CeY_sample_km': ('Ce, Y', 'H = ln(Ce/Y / 0.3029) / 0.0554'),
        }
        _req_inputs, _formula = _ratio_info.get(_primary_proxy_col, ('', ''))
        _ri_c1, _ri_c2 = st.columns(2)
        _ri_c1.info(f'**Required inputs:** {_req_inputs}')
        _ri_c2.info(f'**Formula:** {_formula}')

        # (dataset already selected at the top of the Model tab as primary_source / primary_upload)
        _rat_train_df, _rat_label = read_training_source(primary_source, primary_upload)
        _rat_target = target_col(_rat_train_df) if not _rat_train_df.empty else None

        # Build enriched reference frame and calibration bench
        _rat_enriched = pd.DataFrame()
        _rat_bench = pd.DataFrame()
        if not _rat_train_df.empty:
            _need_game_rat = (_primary_proxy_col == 'H_GAME_LuffiDucea2022_km')
            _rat_enriched = enrich(_rat_train_df.copy(), la_mode, include_game=_need_game_rat)

            # ── GAME sensor subset selection (Luffi & Ducea Table 1 + §5.5.1) ──
            # Only shown when GAME consensus is the active model AND the per-
            # sensor columns exist (i.e. GAME ran successfully). Toggling here
            # post-processes the consensus from existing per-sensor predictions
            # without re-running any LOWESS surfaces.
            if (_need_game_rat
                    and not _rat_enriched.empty
                    and 'GAME_LaYb_H_km' in _rat_enriched.columns):
                _profile = _profile_dataset_mgo(_rat_enriched)
                with st.expander(
                    f'GAME sensor selection — {len(st.session_state.get("game_sensor_subset", list(GAME_SENSOR_META.keys())))} of 41 active',
                    expanded=False,
                ):
                    st.caption(
                        f'Reference dataset MgO: median **{_profile["median"]:.1f}** wt%, '
                        f'p5–p95 **{_profile["p5"]:.1f}–{_profile["p95"]:.1f}**, '
                        f'composition tag: **{_profile["composition"]}** '
                        f'(IQR {_profile["iqr"]:.1f} wt%). '
                        f'Recommendations follow Luffi & Ducea (2022) Table 1 + §5.5.1.'
                    )
                    _preset_c, _apply_c = st.columns([0.7, 0.3])
                    _preset = _preset_c.selectbox(
                        'Sensor preset', GAME_PRESET_LABELS, index=0,
                        key='game_sensor_preset',
                        help=(
                            '**Auto from data** — recommendation based on your MgO '
                            'distribution and element availability.\n\n'
                            '**Top 10 by R²** — paper §5.5.2 endorsed default.\n\n'
                            '**Mafic / Intermediate / Felsic** — sensors whose '
                            'calibrated MgO range covers that lithology band.\n\n'
                            '**MgO-insensitive only** — paper §5.5.1: Sr, Sr/Y, '
                            'Nd/Yb, Sm/Yb, Gd/Yb, Dy/Yb. Safest for mixed datasets.'
                        ),
                    )
                    if _apply_c.button('Apply preset', key='game_apply_preset',
                                       use_container_width=True):
                        _new = get_game_preset(_preset, _rat_enriched)
                        if _new is not None:
                            for _s in GAME_SENSOR_META:
                                st.session_state[f'game_sensor_chk_{_s}'] = (_s in _new)
                            st.session_state['game_sensor_subset'] = list(_new)
                            st.rerun()
                    # Initialise checkbox state on first render — default = all 41
                    if 'game_sensor_subset' not in st.session_state:
                        st.session_state['game_sensor_subset'] = list(GAME_SENSOR_META.keys())
                    for _s in GAME_SENSOR_META:
                        st.session_state.setdefault(
                            f'game_sensor_chk_{_s}',
                            _s in st.session_state['game_sensor_subset'],
                        )
                    # Recommendation tiers for badging
                    _rec = recommend_sensors(_rat_enriched, profile=_profile)
                    _tier = {}
                    for _s in _rec['recommended']:    _tier[_s] = '✓'
                    for _s in _rec['caution']:        _tier[_s] = '⚠'
                    for _s in _rec['not_recommended']: _tier[_s] = '✗'
                    # Render checkboxes in 4 columns sorted by R² (best first)
                    _sorted_sensors = sorted(
                        GAME_SENSOR_META.items(), key=lambda kv: -kv[1][3])
                    _qc1, _qc2, _qc3 = st.columns([0.2, 0.2, 0.6])
                    if _qc1.button('Select all', key='game_sel_all', use_container_width=True):
                        for _s in GAME_SENSOR_META:
                            st.session_state[f'game_sensor_chk_{_s}'] = True
                        st.rerun()
                    if _qc2.button('Clear all', key='game_clear_all', use_container_width=True):
                        for _s in GAME_SENSOR_META:
                            st.session_state[f'game_sensor_chk_{_s}'] = False
                        st.rerun()
                    _qc3.caption(
                        '✓ recommended for this dataset · '
                        '⚠ use with caution · '
                        '✗ not recommended (missing elements or MgO out of range)'
                    )
                    _cols = st.columns(4)
                    for _i, (_s, (_lo, _hi, _diff, _r2, _req)) in enumerate(_sorted_sensors):
                        _badge = _tier.get(_s, '·')
                        _label = f'{_badge} {_s} (R²={_r2:.3f})'
                        _help = (f'MgO range {_lo}–{_hi} wt% · {_diff}-sensitive · '
                                 f'{_rec["reasons"].get(_s, "")}')
                        _cols[_i % 4].checkbox(
                            _label,
                            key=f'game_sensor_chk_{_s}',
                            help=_help,
                        )
                    # Read current subset back from checkbox state
                    _current_subset = [
                        _s for _s in GAME_SENSOR_META
                        if st.session_state.get(f'game_sensor_chk_{_s}', True)
                    ]
                    st.session_state['game_sensor_subset'] = _current_subset
                    if len(_current_subset) < 3:
                        st.warning(
                            f'Only **{len(_current_subset)}** sensor(s) selected — '
                            f'GAME consensus needs at least 3 to produce an estimate.'
                        )
                    elif len(_current_subset) < 41:
                        st.info(
                            f'**{len(_current_subset)} of 41** sensors active. '
                            f'The consensus median, MAD, and CI below are '
                            f'recomputed from this subset only.'
                        )
                # ── Capture all-41 baseline once, then apply user subset ─────
                # By recording the all-41 consensus median up-front, we can
                # show the user how much applying their subset shifted things.
                _baseline = pd.to_numeric(
                    _rat_enriched['H_GAME_LuffiDucea2022_km'], errors='coerce')
                _baseline_n = pd.to_numeric(
                    _rat_enriched.get('GAME_Luffi2022_N_mohometers', np.nan),
                    errors='coerce')
                _rat_enriched = recompute_game_consensus(
                    _rat_enriched,
                    sensor_subset=st.session_state.get(
                        'game_sensor_subset', list(GAME_SENSOR_META.keys())),
                )
                # ── Visible feedback that the consensus was recomputed ──────
                _new_h = pd.to_numeric(
                    _rat_enriched['H_GAME_LuffiDucea2022_km'], errors='coerce')
                _new_n = pd.to_numeric(
                    _rat_enriched.get('GAME_Luffi2022_N_mohometers', np.nan),
                    errors='coerce')
                _delta = (_new_h - _baseline).dropna()
                _gc1, _gc2, _gc3, _gc4 = st.columns(4)
                _gc1.metric(
                    'Active sensors',
                    f'{len(st.session_state.get("game_sensor_subset", []))} / 41',
                )
                _gc2.metric(
                    'Mean firing / sample',
                    f'{_new_n.mean():.1f}' if _new_n.notna().any() else 'n/a',
                    delta=(f'{(_new_n.mean()-_baseline_n.mean()):+.1f}'
                           if _new_n.notna().any() and _baseline_n.notna().any()
                              and abs(_new_n.mean()-_baseline_n.mean()) > 0.05
                           else None),
                )
                _gc3.metric(
                    'Median consensus H',
                    f'{_new_h.median():.1f} km' if _new_h.notna().any() else 'n/a',
                    delta=(f'{(_new_h.median()-_baseline.median()):+.1f} km'
                           if _new_h.notna().any() and _baseline.notna().any()
                              and abs(_new_h.median()-_baseline.median()) > 0.05
                           else None),
                )
                _gc4.metric(
                    'Mean shift vs all-41',
                    f'{_delta.mean():+.2f} km' if len(_delta) > 0 else '0 km',
                    help='Average per-sample change in consensus H when this '
                         'sensor subset is used vs the full 41-sensor consensus. '
                         'Zero = identical predictions; large = subset matters.',
                )

            if _primary_proxy_col in _rat_enriched.columns and _rat_target:
                _rb = _rat_enriched.copy()
                _rb['Predicted_km'] = pd.to_numeric(_rat_enriched[_primary_proxy_col], errors='coerce')
                _rb['Observed_km']  = pd.to_numeric(_rat_enriched[_rat_target], errors='coerce')
                _rb['Residual_km']  = _rb['Predicted_km'] - _rb['Observed_km']
                _rb['Model']        = _ratio_choice
                _rb['Algorithm']    = _pmt
                _rat_bench = _rb.dropna(subset=['Predicted_km', 'Observed_km']).reset_index(drop=True)

        _rat_cov = (int(_rat_enriched[_primary_proxy_col].notna().sum())
                    if _primary_proxy_col in _rat_enriched.columns else 0)
        _rt1, _rt2 = st.columns(2)
        with _rt1:
            summary_tile('Reference dataset', _rat_label)
        with _rt2:
            _cov_label = (f'{_rat_cov:,} / {len(_rat_enriched):,}'
                          if not _rat_enriched.empty else 'no data')
            summary_tile(f'{_pmt} coverage', _cov_label)

        # ── Data Prep unknown-data coverage check ────────────────────────────
        _dp_proc_any = pd.DataFrame()
        for _chk_fi, _chk_f in enumerate(st.session_state.get('dp_global_uploader') or []):
            _chk_proc = st.session_state.get(f'dp_proc_{_chk_fi}_{_chk_f.name}', pd.DataFrame())
            if not _chk_proc.empty:
                _dp_proc_any = _chk_proc
                break
        if not _dp_proc_any.empty:
            _need_game = (_primary_proxy_col == 'H_GAME_LuffiDucea2022_km')
            _enrich_chk = enrich(_dp_proc_any.head(500).copy(), la_mode, include_game=_need_game)
            _cov_n = int(_enrich_chk[_primary_proxy_col].notna().sum()) if _primary_proxy_col in _enrich_chk else 0
            _cov_tot = min(500, len(_dp_proc_any))
            if _cov_n > 0:
                st.success(f'Data Prep preview: **{_cov_n:,}/{_cov_tot:,}** rows have required inputs for _{_ratio_choice}_.')
            else:
                st.warning(f'Data Prep preview: no rows could compute _{_ratio_choice}_ — check that **{_req_inputs}** are mapped.')
        else:
            st.info('Map your data in the **Prepare** tab, then run predictions in the **Predict** tab.')

        # ── Calibration on reference dataset (replaces cross-validation) ─────
        _need_game_rat = (_primary_proxy_col == 'H_GAME_LuffiDucea2022_km')
        with st.expander('Calibration on reference dataset', expanded=True):
            if _rat_bench.empty:
                if _rat_train_df.empty:
                    st.info('Reference dataset not found — add the training file beside the app.')
                elif _rat_cov == 0:
                    st.warning(
                        f'The reference dataset has no rows that can compute '
                        f'**{_ratio_choice}** — required inputs are **{_req_inputs}**.'
                    )
                else:
                    st.info('No rows have both a ratio prediction and a known crustal thickness.')
            else:
                _rcv_a, _rcv_c, _rcv_d = st.columns(3)
                _rcv_color = _rcv_a.selectbox(
                    'Colour by',
                    [c for c in ['Residual_km', 'Age_Ma', 'Rock_Type_Model'] if c in _rat_bench],
                    index=0, key='ratio_cv_color',
                )
                _rcv_size    = int(st.session_state.get('global_point_size', 5))
                _rcv_fit     = _rcv_c.checkbox('Best-fit line', True, key='ratio_cv_fit')
                _rcv_env     = _rcv_d.checkbox('Residual envelope', False, key='ratio_cv_env')
                _r2a, _r2b, _r2c, _r2d = st.columns(4)
                _rcv_pts     = _r2a.checkbox('Show points', True, key='ratio_cv_points')
                _rcv_ma      = _r2b.checkbox('Moving average', False, key='ratio_cv_ma')
                _rcv_ma_type = _r2c.selectbox('Avg type', ['Median', 'Mean'], index=0, key='ratio_cv_ma_type', disabled=not _rcv_ma)
                _rcv_ma_n    = _r2d.slider('Window n', 3, 50, 20, 1, key='ratio_cv_ma_n', disabled=not _rcv_ma)
                st.caption(
                    'How well does this ratio model reproduce known crustal thicknesses '
                    'in the reference dataset? R² and RMSE quantify calibration quality.'
                )

                # ── Side-by-side R²: raw vs. grouped dataset ─────────────────
                # If the pool contains grouped derivatives, pick one and compare.
                _grp_pool_opts = ['(none)'] + list_prepped_dataset_names()
                _grp_cmp_sel = st.selectbox(
                    'Compare R² with grouped dataset',
                    _grp_pool_opts, index=0,
                    key='ratio_r2_compare_pool',
                    help='Pick a dataset from the pool (e.g. one created in the '
                         'Group tab via "Aggregate groups → pool") to see how '
                         'group-median chemistry changes calibration R².',
                )
                _grp_bench_cmp = pd.DataFrame()
                if _grp_cmp_sel != '(none)':
                    _grp_cmp_df = get_prepped_pool().get(_grp_cmp_sel, pd.DataFrame())
                    if not _grp_cmp_df.empty:
                        _grp_cmp_target = target_col(_grp_cmp_df)
                        if _grp_cmp_target:
                            _grp_enr = enrich(_grp_cmp_df.copy(), la_mode,
                                               include_game=_need_game_rat)
                            if _primary_proxy_col in _grp_enr.columns:
                                _grb = _grp_enr.copy()
                                _grb['Predicted_km'] = pd.to_numeric(
                                    _grp_enr[_primary_proxy_col], errors='coerce')
                                _grb['Observed_km']  = pd.to_numeric(
                                    _grp_enr[_grp_cmp_target], errors='coerce')
                                _grb['Residual_km']  = _grb['Predicted_km'] - _grb['Observed_km']
                                _grb['Model']        = _ratio_choice
                                _grb['Algorithm']    = _pmt
                                _grp_bench_cmp = _grb.dropna(
                                    subset=['Predicted_km', 'Observed_km']
                                ).reset_index(drop=True)
                        if _grp_bench_cmp.empty:
                            st.warning(
                                f"'{_grp_cmp_sel}' has no rows with both a ratio "
                                f"prediction and a known thickness column."
                            )

                # Metric strip: raw R² / RMSE and, if available, grouped R² / RMSE
                _raw_summ = display_validation_summary(_rat_bench)
                _raw_r2   = _raw_summ['R2'].iloc[0] if 'R2' in _raw_summ.columns and not _raw_summ.empty else np.nan
                _raw_rmse = _raw_summ['RMSE [Km]'].iloc[0] if 'RMSE [Km]' in _raw_summ.columns and not _raw_summ.empty else np.nan

                if not _grp_bench_cmp.empty:
                    _grp_summ = display_validation_summary(_grp_bench_cmp)
                    _grp_r2   = _grp_summ['R2'].iloc[0] if 'R2' in _grp_summ.columns else np.nan
                    _grp_rmse = _grp_summ['RMSE [Km]'].iloc[0] if 'RMSE [Km]' in _grp_summ.columns else np.nan
                    _m1, _m2, _m3, _m4 = st.columns(4)
                    try:
                        _raw_r2_f  = float(_raw_r2)
                        _grp_r2_f  = float(_grp_r2)
                        _raw_rmse_f = float(_raw_rmse)
                        _grp_rmse_f = float(_grp_rmse)
                        _m1.metric('R² (raw samples)', f'{_raw_r2_f:.3f}')
                        _m2.metric('R² (grouped)',
                                   f'{_grp_r2_f:.3f}',
                                   delta=f'{_grp_r2_f - _raw_r2_f:+.3f}',
                                   help='Positive delta = grouping improves calibration fit. '
                                        'Typical gain +0.3–0.5 when individual-sample '
                                        'scatter is suppressed by per-group medians.')
                        _m3.metric('RMSE km (raw)', f'{_raw_rmse_f:.2f}')
                        _m4.metric('RMSE km (grouped)',
                                   f'{_grp_rmse_f:.2f}',
                                   delta=f'{_grp_rmse_f - _raw_rmse_f:+.2f}',
                                   delta_color='inverse')
                    except (TypeError, ValueError):
                        pass
                    # Show both calibration tables
                    st.markdown(f'**Raw samples** ({len(_rat_bench):,} rows)')
                    st.dataframe(_raw_summ, width='stretch')
                    st.markdown(f'**Grouped** ({len(_grp_bench_cmp):,} groups · {_grp_cmp_sel})')
                    st.dataframe(_grp_summ, width='stretch')
                    _plot1, _plot2 = st.columns(2)
                    with _plot1:
                        st.caption('Raw samples')
                        st.plotly_chart(
                            benchmark_figure(
                                tidy_numbers(_rat_bench), _rcv_size, _rcv_color,
                                _rcv_fit, False, _rcv_env, 'Window', 10.0, 25, False, 5.0,
                                show_moving_avg=_rcv_ma, moving_avg_n=_rcv_ma_n,
                                moving_avg_type=_rcv_ma_type, show_points=_rcv_pts,
                            ),
                            width='stretch', key='ratio_cv_raw_plot',
                        )
                    with _plot2:
                        st.caption(f'Grouped — {_grp_cmp_sel}')
                        st.plotly_chart(
                            benchmark_figure(
                                tidy_numbers(_grp_bench_cmp), _rcv_size, _rcv_color,
                                _rcv_fit, False, _rcv_env, 'Window', 10.0, 25, False, 5.0,
                                show_moving_avg=False, moving_avg_n=_rcv_ma_n,
                                moving_avg_type=_rcv_ma_type, show_points=True,
                            ),
                            width='stretch', key='ratio_cv_grp_plot',
                        )
                else:
                    # No comparison — single table + plot
                    st.dataframe(_raw_summ, width='stretch')
                    st.plotly_chart(
                        benchmark_figure(
                            tidy_numbers(_rat_bench), _rcv_size, _rcv_color,
                            _rcv_fit, False, _rcv_env, 'Window', 10.0, 25, False, 5.0,
                            show_moving_avg=_rcv_ma, moving_avg_n=_rcv_ma_n,
                            moving_avg_type=_rcv_ma_type, show_points=_rcv_pts,
                        ),
                        width='stretch',
                    )
        # ── Feature importance / GAME mohometer coverage ─────────────────────
        _fi_expander_title = (
            'GAME mohometer coverage'
            if _primary_proxy_col == 'H_GAME_LuffiDucea2022_km'
            else 'Feature importance'
        )
        with st.expander(_fi_expander_title, expanded=False):
            if _primary_proxy_col == 'H_GAME_LuffiDucea2022_km':
                _game_h_cols = sorted([
                    c for c in _rat_enriched.columns
                    if c.startswith('GAME_') and c.endswith('_H_km')
                ])
                if _game_h_cols and not _rat_enriched.empty:
                    _sensor_rows = []
                    for _sc in _game_h_cols:
                        _sname = _sc.replace('GAME_', '').replace('_H_km', '')
                        _svals = pd.to_numeric(_rat_enriched[_sc], errors='coerce').dropna()
                        _meta = GAME_SENSOR_META.get(_sname, (np.nan, np.nan, '', np.nan, ()))
                        _sensor_rows.append({
                            'Mohometer': _sname,
                            'Coverage_%': round(float(_rat_enriched[_sc].notna().mean() * 100), 1),
                            'N_samples': int(_rat_enriched[_sc].notna().sum()),
                            'Median_H_km': round(float(_svals.median()), 1) if len(_svals) > 0 else np.nan,
                            'R2': float(_meta[3]) if np.isfinite(_meta[3]) else np.nan,
                            'MgO_range': f'{_meta[0]}–{_meta[1]} wt%' if np.isfinite(_meta[0]) else '',
                            'Diff_class': _meta[2] or '',
                        })
                    _sensor_df = pd.DataFrame(_sensor_rows)
                    # Sort + colour controls
                    _bc1, _bc2 = st.columns([0.5, 0.5])
                    _sort_by = _bc1.selectbox(
                        'Order bars by',
                        ['R² (best at top)', 'Coverage (highest at top)',
                         'Coverage (lowest at top)', 'Mohometer name'],
                        index=0, key='game_coverage_sort',
                    )
                    _color_by = _bc2.selectbox(
                        'Colour bars by',
                        ['R²', 'Coverage', 'Differentiation class'],
                        index=0, key='game_coverage_color',
                    )
                    if _sort_by.startswith('R²'):
                        _sensor_df = _sensor_df.sort_values('R2', ascending=True, na_position='first')
                    elif _sort_by == 'Coverage (highest at top)':
                        _sensor_df = _sensor_df.sort_values('Coverage_%', ascending=True)
                    elif _sort_by == 'Coverage (lowest at top)':
                        _sensor_df = _sensor_df.sort_values('Coverage_%', ascending=False)
                    else:
                        _sensor_df = _sensor_df.sort_values('Mohometer', ascending=False)
                    st.caption(
                        f'{len(_game_h_cols)} GAME mohometers across '
                        f'{len(_rat_enriched):,} reference samples — '
                        f'bar length = sample fraction the sensor could compute; '
                        f'colour = {_color_by.lower()} (Luffi & Ducea 2022 Table 1).'
                    )
                    # Diverging red→cream→blue scale used for both R² and Coverage
                    # so high values (good) read blue and low values read red.
                    _diverging_scale = [[0.0,'#8f1729'], [0.48,'#f4dfcf'],
                                        [0.50,'#f7f7f4'], [0.52,'#d9e7f0'],
                                        [1.0,'#244575']]
                    if _color_by == 'R²':
                        _fi_fig = px.bar(
                            _sensor_df, y='Mohometer', x='Coverage_%', orientation='h',
                            color='R2', color_continuous_scale=_diverging_scale,
                            range_color=[0.91, 0.98],
                            labels={'Coverage_%':'Coverage (%)','Mohometer':'GAME sensor','R2':'R²'},
                            hover_data=['N_samples','Median_H_km','R2','MgO_range','Diff_class'],
                            template='plotly_white',
                        )
                        _fi_fig.update_layout(coloraxis_showscale=True,
                                              coloraxis_colorbar=dict(title='R²', len=0.6))
                    elif _color_by == 'Coverage':
                        _fi_fig = px.bar(
                            _sensor_df, y='Mohometer', x='Coverage_%', orientation='h',
                            color='Coverage_%', color_continuous_scale=_diverging_scale,
                            range_color=[0, 100],
                            labels={'Coverage_%':'Coverage (%)','Mohometer':'GAME sensor'},
                            hover_data=['N_samples','Median_H_km','R2','MgO_range','Diff_class'],
                            template='plotly_white',
                        )
                        _fi_fig.update_layout(coloraxis_showscale=True,
                                              coloraxis_colorbar=dict(title='Coverage (%)', len=0.6))
                    else:  # Differentiation class — editorial palette matching app style
                        _class_colors = {
                            'insensitive': '#3a6ea5',  # steel blue (calm, reliable)
                            'mild':        '#c8a87e',  # warm sand (neutral)
                            'strong':      '#a83a3a',  # terracotta (warning)
                            '':            '#9aa0a6',  # neutral grey
                        }
                        _fi_fig = px.bar(
                            _sensor_df, y='Mohometer', x='Coverage_%', orientation='h',
                            color='Diff_class', color_discrete_map=_class_colors,
                            category_orders={'Diff_class':['insensitive','mild','strong','']},
                            labels={'Coverage_%':'Coverage (%)','Mohometer':'GAME sensor',
                                    'Diff_class':'MgO sensitivity'},
                            hover_data=['N_samples','Median_H_km','R2','MgO_range'],
                            template='plotly_white',
                        )
                        _fi_fig.update_layout(legend=dict(orientation='h', y=-0.1))
                    _fi_fig.update_layout(
                        height=max(260, len(_game_h_cols) * 22),
                        margin=dict(l=10, r=10, t=20, b=10),
                    )
                    st.plotly_chart(_fi_fig, width='stretch', key='game_sensor_coverage_fig')

                    # ── Click-to-inspect: per-sensor LOWESS calibration surface ──
                    _inspect_opts = sorted(
                        [s for s in GAME_SENSOR_META if s in game_calibrators()],
                        key=lambda s: -GAME_SENSOR_META[s][3],
                    )
                    if _inspect_opts:
                        _ic1, _ic2 = st.columns([0.7, 0.3])
                        _inspect_sensor = _ic1.selectbox(
                            'Inspect calibration surface for sensor',
                            _inspect_opts,
                            index=0,
                            key='game_inspect_sensor',
                            format_func=lambda s: (
                                f'{s} (R²={GAME_SENSOR_META[s][3]:.3f}, '
                                f'MgO {GAME_SENSOR_META[s][0]}–{GAME_SENSOR_META[s][1]}, '
                                f'{GAME_SENSOR_META[s][2]}-sensitive)'
                            ),
                            help='Shows the LOWESS calibration surface (Moho depth as a '
                                 'function of MgO and the sensor variable) plus the '
                                 'arc median datapoints used to fit it.',
                        )
                        _active = st.session_state.get('game_sensor_subset',
                                                       list(GAME_SENSOR_META.keys()))
                        _ic2.metric(
                            'Active in consensus?',
                            'Yes' if _inspect_sensor in _active else 'No (toggled off)',
                        )
                        _cal_fig = game_calibration_figure(_rat_enriched, _inspect_sensor)
                        if _cal_fig is not None:
                            st.plotly_chart(_cal_fig, width='stretch',
                                            key=f'game_cal_{_inspect_sensor}')
                        else:
                            st.info(f'No calibration surface available for {_inspect_sensor}.')
                else:
                    # Read the status column that apply_luffi_game_mohometers always writes
                    _game_statuses = set(
                        _rat_enriched['GAME_Luffi2022_Status'].dropna().unique()
                        if 'GAME_Luffi2022_Status' in _rat_enriched.columns
                        else []
                    )
                    if _rat_enriched.empty:
                        st.info('Load a reference dataset to see GAME mohometer coverage.')
                    elif any('unavailable' in str(s).lower() for s in _game_statuses):
                        st.warning(
                            f'The GAME calibration file (**{GAME_CALIBRATION_FILE}**) '
                            f'was not found beside the app. '
                            f'Place the Luffi & Ducea (2022) arc calibration CSV '
                            f'in the same folder as Mohometer.py and reload — '
                            f'GAME cannot run without it.'
                        )
                    elif any('missing mgo' in str(s).lower() for s in _game_statuses):
                        st.warning(
                            'GAME requires an **MgO** column — it was not found '
                            'in the reference dataset.'
                        )
                    else:
                        _diag_df = game_sensor_diagnosis(_rat_enriched)
                        _n_fired      = (_diag_df['Status'] == 'Fired').sum()
                        _n_no_cal     = (_diag_df['Status'] == 'Not in calibration file').sum()
                        _n_no_data    = (_diag_df['Status'] == 'No element data').sum()
                        _n_outside    = (_diag_df['Status'] == 'Outside calibration domain').sum()
                        _n_skipped    = _diag_df['Status'].str.startswith('Skipped').sum()
                        if _n_outside > 0 and _n_no_data == len(_diag_df) - _n_no_cal - _n_skipped - _n_outside:
                            st.warning(
                                f'**{_n_outside}** sensor(s) had element data but all samples fell '
                                f'**outside the LOESS calibration domain** — no finite estimates '
                                f'could be produced. The reference dataset\'s MgO vs element-ratio '
                                f'values may not overlap with the Luffi & Ducea (2022) arc training data.'
                            )
                        elif _n_no_data >= len(_diag_df) - _n_no_cal - _n_skipped:
                            st.warning(
                                f'No GAME sensors could find the required element columns in this '
                                f'reference dataset. **{_n_no_cal}** sensors have no calibration; '
                                f'**{_n_no_data}** sensors are missing element columns (La, Yb, Ce, '
                                f'Nd, Y, Nb, Sr, Ba, Th…). Upload a dataset with trace elements to '
                                f'enable GAME coverage.'
                            )
                        else:
                            st.warning(
                                f'No GAME sensors produced finite estimates. '
                                f'**{_n_no_cal}** not calibrated · '
                                f'**{_n_no_data}** missing elements · '
                                f'**{_n_outside}** outside calibration domain · '
                                f'**{_n_skipped}** skipped (high elevation RMSE).'
                            )
                        with st.expander('Sensor diagnostic detail', expanded=False):
                            _status_order = ['Fired','Outside calibration domain',
                                             'No element data','Skipped','Not in calibration file']
                            _diag_show = _diag_df.copy()
                            _diag_show['_sort'] = _diag_show['Status'].apply(
                                lambda s: next((i for i,k in enumerate(_status_order) if s.startswith(k)), 99))
                            st.dataframe(
                                _diag_show.sort_values('_sort').drop(columns='_sort').rename(
                                    columns={'Sensor':'GAME sensor','Status':'Reason',
                                             'N_samples_with_data':'N with data'}),
                                hide_index=True, width='stretch',
                            )
            elif _primary_proxy_col == 'H_Sundell2021_Paired_km':
                st.info(
                    '**Sundell et al. (2021) paired model** combines two ratios with '
                    'fixed published weights:\n\n'
                    'H = **10.3** × ln(Sr/Y) + **8.8** × ln(La/Yb(N)) − 10.6\n\n'
                    'Both inputs carry fixed coefficients — there is no data-driven '
                    'importance to display.'
                )
            else:
                st.info(
                    'Single-ratio models use a single element ratio — '
                    'there is no multi-feature decomposition to show.'
                )

        # ── Reference dataset map ─────────────────────────────────────────────
        # Gate same as the Training sample map (st.expander is visual only —
        # its body re-runs on every Streamlit rerun regardless of the open
        # state and regardless of which tab is active).
        with st.expander('Reference dataset map', expanded=False):
            _rm_render = st.checkbox(
                'Render map',
                value=False,
                key='ratio_ref_map_render',
                help='Off by default — rebuilds the geo-scatter (plus optional '
                     'CRUST1.0 / LithoRef18 background) on every Streamlit '
                     'rerun. Tick to render.',
            )
            _rm_src = _rat_enriched if not _rat_enriched.empty else pd.DataFrame()
            if not _rm_render:
                st.caption('🗺️ Tick **Render map** above to compute and display the reference dataset map.')
            elif {'Lat', 'Lon'}.issubset(_rm_src):
                _rm_color_opts = []
                if _primary_proxy_col in _rm_src:
                    _rm_color_opts.append(_primary_proxy_col)
                for _rmc in ['GAME_Luffi2022_N_mohometers', 'Crust_Thickness', 'Age_Ma',
                             'Geologic_Era', 'Geologic_Period', 'Geologic_Epoch',
                             'Geologic_Age_Label', 'Tectonic_Setting', 'Arc_or_Segment',
                             'Geologic_Domain', 'Dataset', 'Rock_Type_Model']:
                    if _rmc in _rm_src and _rmc not in _rm_color_opts:
                        _rm_color_opts.append(_rmc)
                if _rm_color_opts:
                    _rmc1, _rmc2, _rmc3, _rmc4 = st.columns([1.1, 1, 0.9, 0.6])
                    _rm_crust1   = read_default_crust_grid(str(DEFAULT_CRUST1_GRID))
                    _rm_lithoref = read_default_lithoref18()
                    _rm_bg_choices = ['None']
                    if not _rm_crust1.empty:
                        _rm_bg_choices.insert(0, 'CRUST1.0')
                    if not _rm_lithoref.empty:
                        _rm_bg_choices.insert(min(1, len(_rm_bg_choices)), 'LithoRef18 / Alfonso 2019')
                    _rm_bg      = _rmc1.selectbox('Background grid', _rm_bg_choices, index=0, key='ratio_map_bg_grid')
                    _rm_bg_mode = _rmc1.radio('Display as', ['Surface', 'Points'], horizontal=True, key='ratio_map_bg_mode')
                    normalize_widget_state('ratio_map_layer', _rm_color_opts)
                    _rm_color_by = _rmc2.selectbox(
                        'Sample layer', _rm_color_opts, index=0, key='ratio_map_layer',
                        format_func=lambda c: PROXY_THICKNESS_LABELS.get(c, c),
                    )
                    _rm_zoom = _rmc4.checkbox('Zoom to data', False, key='ratio_map_zoom')
                    _rm_df = plot_df(_rm_src.dropna(subset=['Lat', 'Lon']))
                    _rm_color_by = resolve_option(_rm_color_by, list(_rm_df.columns), 0)
                    _rm_geo = geo_downsample(_rm_df)
                    if len(_rm_geo) < len(_rm_df):
                        _rmc3.caption(f'{len(_rm_geo):,} / {len(_rm_df):,} pts')
                    _rm_hover = hover_cols(
                        _rm_geo,
                        ['Sample_ID', 'Age_Ma', 'Geologic_Era', 'Geologic_Period',
                         'Tectonic_Setting', 'Arc_or_Segment', 'Geologic_Domain',
                         'Dataset', 'Rock_Type_Model', 'Crust_Thickness',
                         'GAME_Luffi2022_N_mohometers'],
                        _rm_color_by, 'Lat', 'Lon',
                    )
                    _rm_fig = px.scatter_geo(
                        _rm_geo, lat='Lat', lon='Lon', color=_rm_color_by,
                        hover_data=map_hover_data(_rm_geo, _rm_hover),
                        projection='natural earth', template='plotly_white',
                        labels={_rm_color_by: PROXY_THICKNESS_LABELS.get(_rm_color_by, _rm_color_by)},
                        **map_color_kw(_rm_color_by, _rm_geo),
                    )
                    _rm_fig.update_traces(
                        marker=dict(line=dict(color='black', width=0.7)),
                        selector=dict(type='scattergeo'),
                    )
                    if _rm_bg == 'CRUST1.0' and not _rm_crust1.empty:
                        _rm_fig = add_grid_background_to_geofig(
                            _rm_fig, _rm_crust1, 'CRUST1_Total_Crust_km',
                            'CRUST1.0', 'CRUST1.0', _rm_bg_mode,
                        )
                    elif _rm_bg == 'LithoRef18 / Alfonso 2019' and not _rm_lithoref.empty:
                        _rm_fig = add_grid_background_to_geofig(
                            _rm_fig, _rm_lithoref, 'LithoRef18_Total_Crust_km',
                            'LithoRef18 / Alfonso 2019', 'LithoRef18', _rm_bg_mode,
                        )
                    _rm_fig = style_training_map_legends(_rm_fig, _rm_color_by)
                    _rm_fig = apply_geo_zoom_to_data(_rm_fig, _rm_df, _rm_zoom)
                    _rm_fig.update_layout(height=650)
                    st.plotly_chart(_rm_fig, width='stretch', key='ratio_ref_map_fig')
                    if _rmc3.checkbox('Coverage density', False, key='ratio_map_density'):
                        _rm_lats = pd.to_numeric(_rm_df['Lat'], errors='coerce').dropna()
                        _rm_lons = pd.to_numeric(_rm_df['Lon'], errors='coerce').dropna()
                        if len(_rm_lats) >= 10:
                            _rm_deg = 10
                            _rm_lat_bins = np.arange(-90, 91, _rm_deg)
                            _rm_lon_bins = np.arange(-180, 181, _rm_deg)
                            _rm_hist, _, _ = np.histogram2d(_rm_lats, _rm_lons, bins=[_rm_lat_bins, _rm_lon_bins])
                            _rm_clat = ((_rm_lat_bins[:-1] + _rm_lat_bins[1:]) / 2)
                            _rm_clon = ((_rm_lon_bins[:-1] + _rm_lon_bins[1:]) / 2)
                            _rm_clon2, _rm_clat2 = np.meshgrid(_rm_clon, _rm_clat)
                            _rm_dens = pd.DataFrame({'lat': _rm_clat2.ravel(), 'lon': _rm_clon2.ravel(), 'n': _rm_hist.ravel()})
                            _rm_dens = _rm_dens[_rm_dens['n'] == 0].copy()
                            if not _rm_dens.empty:
                                st.caption(f'**{len(_rm_dens)} empty {_rm_deg}° cells** — predictions into these regions are extrapolations.')
                                _rm_gap_fig = px.scatter_geo(_rm_dens, lat='lat', lon='lon',
                                                             projection='natural earth', template='plotly_white',
                                                             color_discrete_sequence=['#ef4444'])
                                _rm_gap_fig.update_traces(marker=dict(size=8, symbol='square', opacity=0.4,
                                                                      line=dict(color='#b91c1c', width=0.5)))
                                _rm_gap_fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                                st.plotly_chart(_rm_gap_fig, width='stretch', key='ratio_coverage_gaps')
                            else:
                                st.success(f'All {_rm_deg}° cells have at least one reference sample.')
                else:
                    st.info('No colour columns found for the reference map.')
            else:
                st.info('Reference dataset map needs Lat and Lon columns.')

    # ── Machine Learning panel ───────────────────────────────────────────────
    if _pmt_is_ml:
        st.session_state.pop('_primary_proxy_col', None)
        st.session_state.pop('_primary_proxy_name', None)
        st.header('Machine Learning Model')
        # (dataset already selected at the top of the Model tab as primary_source / primary_upload)
        c1,c2=st.columns([1,2])
        try:
            train_df,source_label=read_training_source(primary_source,primary_upload)
            if not train_df.empty:
                train_df,target=target_column_controls('Primary model',train_df)
                if target is None:
                    st.stop()
                primary_train_df=training_subset_controls('Primary model',train_df,expanded=False)
                if primary_train_df.empty:
                    st.warning(
                        'All training rows were filtered out — check the age range, '
                        'crustal-thickness range, and rock-type filters in the '
                        '"Primary model training filters" expander above.'
                    )
                    raise StopIteration  # jump to except handler silently
                # Granularity toggle — only renders when training_df has Group_ID.
                # Applies to both Single and Multi model approaches.
                primary_train_df, _train_granularity = dataset_granularity_controls(
                    primary_train_df, 'primary_train', label='Training data',
                )
                default_set = ('Zou et al. (2021)' if primary_source == 'Zou et al. (2021)'
                               else 'Luffi & Ducea (2022)' if primary_source == 'Luffi & Ducea (2022)'
                               else 'Guo & Yang (2023)' if primary_source == 'Guo & Yang (2023)'
                               else 'Full suite')
                # Feature importance — computed on the full filtered training set
                _guo_all = FEATURE_SETS['Guo & Yang (2023)']
                _guo_present = [f for f in _guo_all if f in primary_train_df]
                if _guo_present:
                    _guo_num = primary_train_df[_guo_present + [target]].apply(pd.to_numeric, errors='coerce')
                    _guo_complete_n = int(_guo_num.notna().all(axis=1).sum())
                    _feat_cov = _guo_num[_guo_present].notna().mean()
                else:
                    _guo_complete_n = 0; _feat_cov = pd.Series(dtype=float)
                if len(_guo_present) == len(_guo_all) and _guo_complete_n > 0:
                    importance_features = _guo_all
                elif _guo_present:
                    importance_features = [f for f in _guo_present if _feat_cov.get(f, 0) >= 0.25] or _guo_present
                else:
                    importance_features = dataset_numeric_features(primary_train_df, target)
                full_model, full_clean = train_model(primary_train_df,target,importance_features,seed,'ExtraTrees')
                full_importance = feature_importance(full_model,importance_features, df=full_clean, target=target)
                configs=[]

                # ── Summary tiles (shown in all approaches) ──────────────────
                with c2:
                    m1,m2=st.columns(2)
                    with m1: summary_tile('Training source',source_label)
                    with m2: summary_tile('Training rows',len(full_clean),numeric=True)

                # (_ml_approach already set at the very top of the Model tab)

                # ── SINGLE MODEL ─────────────────────────────────────────────
                if _ml_approach == 'Single model':
                    with c1:
                        default_algorithm = 'XGBoost' if primary_source == 'Zou et al. (2021)' and 'XGBoost' in ALGORITHMS else 'ExtraTrees'
                        normalize_widget_state('primary_algorithm',ALGORITHMS)
                        primary_algorithm=st.selectbox('4. ML algorithm',ALGORITHMS,index=ALGORITHMS.index(default_algorithm) if default_algorithm in ALGORITHMS else 0,key='primary_algorithm')
                    primary_feature_set,primary_features=feature_strategy_controls('Primary model',primary_train_df,target,default_set,full_importance,collapsed=False,allow_min_ri=False,expander_label='4. Element list')
                    if primary_features:
                        default_primary_name=f'Primary: {primary_feature_set} / {primary_algorithm}'
                        normalize_widget_state('primary_model_name')
                        primary_model_name=st.text_input('Model name',value=default_primary_name,key='primary_model_name')
                        primary_model_name=display_algorithm_label(primary_model_name)
                        configs.append({'label':primary_model_name.strip() or default_primary_name,'df':primary_train_df,'target':target,'features':primary_features,'feature_set':primary_feature_set,'algorithm':primary_algorithm,'source':primary_source})
                    with st.expander('Compare models',expanded=False):
                        add_model2=st.checkbox('Add comparison model',False)
                        if add_model2:
                            model2_kind=st.radio('Model 2 type',['Primary model - modified element list','New model'],horizontal=True,key='model2_kind')
                            if model2_kind == 'Primary model - modified element list':
                                model2_source=primary_source; model2_upload=primary_upload; model2_df=primary_train_df; model2_label=source_label; model2_target=target; model2_algorithm=primary_algorithm; model2_default=default_set
                            else:
                                model2_source=st.selectbox('Model 2 training model',source_options,index=0,key='model2_source')
                                model2_upload=None
                                if model2_source == 'Upload dataset':
                                    model2_upload=st.file_uploader('Upload Model 2 training dataset',type=['csv','xlsx','xls'],key='model2_training_dataset')
                                model2_df,model2_label=read_training_source(model2_source,model2_upload)
                                if not model2_df.empty:
                                    model2_df,model2_target=target_column_controls('Model 2',model2_df)
                                    model2_df=training_subset_controls('Model 2',model2_df,expanded=False)
                                else:
                                    model2_target=None
                                model2_default = ('Zou et al. (2021)' if model2_source == 'Zou et al. (2021)'
                                                  else 'Luffi & Ducea (2022)' if model2_source == 'Luffi & Ducea (2022)'
                                                  else 'Guo & Yang (2023)' if model2_source == 'Guo & Yang (2023)'
                                                  else 'Full suite')
                                model2_default_algorithm = 'XGBoost' if model2_source == 'Zou et al. (2021)' and 'XGBoost' in ALGORITHMS else primary_algorithm
                                normalize_widget_state('model2_algorithm',ALGORITHMS)
                                model2_algorithm=st.selectbox('Model 2 ML method',ALGORITHMS,index=ALGORITHMS.index(model2_default_algorithm) if model2_default_algorithm in ALGORITHMS else 0,key='model2_algorithm')
                            if model2_df.empty:
                                st.info('Model 2 needs a training dataset before it can be compared.')
                            else:
                                allow_ri = model2_source == primary_source and model2_algorithm == primary_algorithm
                                if not allow_ri:
                                    st.caption('Minimum RI is available only when Model 2 uses the same training model and ML method as the primary model.')
                                model2_feature_set,model2_features=feature_strategy_controls('Model 2',model2_df,model2_target,model2_default,full_importance,collapsed=False,use_expander=False,allow_min_ri=allow_ri)
                                if model2_features:
                                    label_prefix='Modified' if model2_kind == 'Primary model - modified element list' else 'Model 2'
                                    default_model2_name=f'{label_prefix}: {model2_feature_set} / {model2_algorithm}'
                                    normalize_widget_state('model2_name')
                                    model2_name=st.text_input('Model 2 name',value=default_model2_name,key='model2_name')
                                    model2_name=display_algorithm_label(model2_name)
                                    configs.append({'label':model2_name.strip() or default_model2_name,'df':model2_df,'target':model2_target,'features':model2_features,'feature_set':model2_feature_set,'algorithm':model2_algorithm,'source':model2_source})

                # ── MULTI-MODEL (per group) ───────────────────────────────────
                else:
                    with c1:
                        default_algorithm = 'XGBoost' if primary_source == 'Zou et al. (2021)' and 'XGBoost' in ALGORITHMS else 'ExtraTrees'
                        normalize_widget_state('mm_default_alg', ALGORITHMS)
                        _mm_default_alg = st.selectbox(
                            'Default algorithm (for new slots)', ALGORITHMS,
                            index=ALGORITHMS.index(default_algorithm) if default_algorithm in ALGORITHMS else 0,
                            key='mm_default_alg',
                        )
                    _multi_model_slots_ui(
                        primary_train_df, target, full_importance,
                        default_set, primary_source, _mm_default_alg, configs,
                    )
                # ── Training is now explicit: build configs interactively, then
                # press the Train button to actually fit. Until pressed, the
                # tab reads the previously-trained models from session state
                # so model importance / cross-validation panels stay visible.
                # When the config signature changes vs. the last training, a
                # banner prompts the user to retrain.
                import hashlib as _hl_train
                _train_btn_c1, _train_btn_c2 = st.columns([1, 4])
                # Compute a config signature so we can detect "out of date"
                _cfg_sig_payload = json.dumps([
                    {
                        'label':       str(cfg.get('label')),
                        'algorithm':   str(cfg.get('algorithm')),
                        'feature_set': str(cfg.get('feature_set')),
                        'features':    list(cfg.get('features', [])),
                        'target':      str(cfg.get('target')),
                        'source':      str(cfg.get('source')),
                        'rows':        int(len(cfg.get('df', pd.DataFrame()))),
                    }
                    for cfg in configs
                ], sort_keys=True)
                _cfg_sig = _hl_train.md5((_cfg_sig_payload + f'|seed={seed}').encode('utf-8')).hexdigest()
                _trained_sig = st.session_state.get('_models_trained_sig')
                _trained_models  = st.session_state.get('_models_trained_models', {})
                _trained_valdf   = st.session_state.get('_models_trained_validation', pd.DataFrame())
                _trained_impdf   = st.session_state.get('_models_trained_importance', pd.DataFrame())
                _is_stale = (_trained_sig != _cfg_sig) and bool(configs)

                with _train_btn_c1:
                    if st.button('▶ Train models',
                                 key='train_models_btn',
                                 use_container_width=True,
                                 type='primary' if _is_stale or not _trained_models else 'secondary',
                                 disabled=not configs,
                                 help='Run training for every model slot above. The result is '
                                      'cached until you change the config — Validate / Predict / '
                                      'plots all read from the saved snapshot.'):
                        _new_models, _new_valdf, _new_impdf = train_configured_models(configs, seed)
                        st.session_state['_models_trained_models']     = _new_models
                        st.session_state['_models_trained_validation'] = _new_valdf
                        st.session_state['_models_trained_importance'] = _new_impdf
                        st.session_state['_models_trained_sig']        = _cfg_sig
                        if not _new_impdf.empty:
                            st.session_state['_cached_importance_df'] = _new_impdf.copy()
                        st.session_state['_training_sources_used']    = {c.get('source','') for c in configs}
                        st.rerun()
                with _train_btn_c2:
                    if not configs:
                        st.caption('Configure at least one model slot above before training.')
                    elif not _trained_models:
                        st.caption('No models trained yet — press **▶ Train models** to fit.')
                    elif _is_stale:
                        st.warning(
                            '⚠ Config changed since the last training run. Press '
                            '**▶ Train models** to refresh — Validate, Predict, and the '
                            'plots below still show the previous run\'s results.'
                        )
                    else:
                        st.caption(f'✓ Showing the last training run ({len(_trained_models)} model(s)).')

                # Read trained results from session state (or empty if never trained)
                models         = _trained_models if _trained_models else {}
                validation_df  = _trained_valdf  if not _trained_valdf.empty  else pd.DataFrame()
                importance_df  = _trained_impdf  if not _trained_impdf.empty  else pd.DataFrame()

                primary_label = display_algorithm_label(configs[0]['label']) if configs else ''
                if primary_label and primary_label in models:
                    model=models[primary_label]['model']; clean=models[primary_label]['clean']; selected_features=models[primary_label]['features']; feature_set_name=primary_label
                with c2:
                    summary_tile('Models',len(models),numeric=True,compact=True)
                if not models:
                    # Skip the rest of the model-tab UI until at least one
                    # training run has been recorded — saves a wall of
                    # widgets / charts that have nothing to plot.
                    raise StopIteration
                st.divider()
                model_options=list(models.keys())
                normalize_widget_state('active_interp_model',model_options)
                active_model_label=st.selectbox(
                    'Primary model',
                    model_options, index=0, key='active_interp_model',
                    help='Headline ML model. Drives the Model-tab diagnostics '
                         '(feature importance, partial dependence) AND becomes '
                         'the default Y axis in the Validate / Predict tabs\' '
                         'Validation plot. Each trained model also gets its '
                         'own per-model column (Predicted_<name>_km / '
                         'Residual_<name>_km / CI_Low_<name>_km / CI_High_<name>_km) '
                         'so you can pick a different model on either axis there.',
                )
                if active_model_label in models:
                    model=models[active_model_label]['model']; clean=models[active_model_label]['clean']; selected_features=models[active_model_label]['features']; feature_set_name=active_model_label
                if st.button('Save selected model as preset',key='save_model_preset'):
                    preset_path=Path('saved_model_presets.csv')
                    if active_model_label in models:
                        row=pd.DataFrame([{'Preset':active_model_label,'Model':active_model_label,'Feature_Set':models[active_model_label]['feature_set'],'Algorithm':models[active_model_label]['algorithm'],'Features':', '.join(models[active_model_label]['features'])}])
                        existing=pd.read_csv(preset_path) if preset_path.exists() else pd.DataFrame()
                        pd.concat([existing,row],ignore_index=True).drop_duplicates(subset=['Preset'],keep='last').to_csv(preset_path,index=False)
                        st.success(f'Saved preset metadata to {preset_path}')
            else:
                with c1:
                    st.info('Primary model controls appear after a training model is loaded.')
                st.warning('Add the Guo & Yang 2023 training workbook beside the app, or upload a training dataset here.')
        except StopIteration:
            pass  # warning already shown above (filtered-out rows)
        except Exception as e:
            _e_msg = str(e)
            if 'No complete training rows' in _e_msg or 'no complete' in _e_msg.lower():
                st.warning(
                    'No complete rows found for the selected features after numeric '
                    'coercion.  Try: (1) click **↻ Reload** to force a fresh file read, '
                    '(2) switch to a different training source, or (3) choose a different '
                    f'element set.  Detail: _{_e_msg}_'
                )
            elif 'StopException' in type(e).__name__:
                pass  # st.stop() was called inside; let Streamlit handle it
            else:
                st.error(f'Model setup failed: {_e_msg}')
    
        with st.expander('Model cross-validation',expanded=True):
            if validation_df.empty:
                st.info('Cross-validation appears after the model is trained.')
            else:
                cv_models=list(validation_df['Model'].dropna().astype(str).unique())
                normalize_widget_state('model_cv_selected',cv_models)
                cv_selected=st.multiselect('Models',cv_models,default=cv_models[:min(len(cv_models),3)],key='model_cv_selected')
                cv_plot=validation_df[validation_df['Model'].astype(str).isin(cv_selected)].copy() if cv_selected else validation_df.copy()
                if cv_plot.empty:
                    st.info('Select at least one model to show the cross-validation graph.')
                else:
                    cva,cvc,cvd=st.columns(3)
                    cv_color=cva.selectbox('Colour points by',['Delta_km','Model','Algorithm'],index=0,key='model_cv_color',format_func=lambda c: 'model - known [Km]' if c == 'Delta_km' else c)
                    cv_size=int(st.session_state.get('global_point_size', 5))
                    cv_fit=cvc.checkbox('Best-fit lines',True,key='model_cv_best_fit')
                    cv_envelope=cvd.checkbox('Best-fit residual envelopes',False,key='model_cv_envelope')
                    cv_r2a,cv_r2b,cv_r2c,cv_r2d=st.columns(4)
                    cv_pts=cv_r2a.checkbox('Show points',True,key='model_cv_points')
                    cv_ma=cv_r2b.checkbox('Moving average',False,key='model_cv_ma')
                    cv_ma_type=cv_r2c.selectbox('Avg type',['Median','Mean'],index=0,key='model_cv_ma_type',disabled=not cv_ma)
                    cv_ma_n=cv_r2d.slider('Window n',3,50,20,1,key='model_cv_ma_n',disabled=not cv_ma)
                    st.caption('R2 shows fit quality. RMSE and MAE are average prediction error in Km. Bias is mean model - known thickness; positive means the model overestimates.')
                    st.dataframe(display_validation_summary(cv_plot),width='stretch')
                    st.plotly_chart(benchmark_figure(tidy_numbers(cv_plot),cv_size,cv_color,cv_fit,False,cv_envelope,'Window',10.0,25,False,5.0,show_moving_avg=cv_ma,moving_avg_n=cv_ma_n,moving_avg_type=cv_ma_type,show_points=cv_pts),width='stretch')
    
        with st.expander('Feature importance',expanded=False):
            _render_feature_importance_panel(importance_df, list(models.keys()),
                                             'importance_model', 'importance_sort')
    
        with st.expander('Training sample map',expanded=False):
            # Gate the heavy work (enrich + per-model predict over the full
            # training set, plus CRUST1.0 / LithoRef18 grid loads) behind a
            # checkbox. Default OFF because Streamlit re-executes every tab
            # body on every interaction — without this gate the map would
            # rebuild on every widget change anywhere in the app.
            _render_tm = st.checkbox(
                'Render map',
                value=False,
                key='training_map_render',
                help='Off by default — rendering the map runs enrich + per-model '
                     'predict over the full training set and loads the CRUST1.0 / '
                     'LithoRef18 background grids. That work runs every Streamlit '
                     'rerun, so leaving this off avoids paying the cost when you '
                     'are working elsewhere in the app. Tick when you want to view '
                     'the map.',
            )
            if not _render_tm:
                st.caption('🗺️ Tick **Render map** above to compute and display the training-sample map.')
            else:
                map_source_df = primary_train_df if 'primary_train_df' in locals() and not primary_train_df.empty else train_df
                if {'Lat','Lon'}.issubset(map_source_df) and not map_source_df.empty:
                    map_df=enrich(map_source_df.copy(),la_mode)
                    color_options=[c for c in ['Age_Ma','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type_Model'] if c in map_df]
                    modelled_cols=[]
                    for name,bundle in models.items():
                        ok,_=complete(map_df,bundle['features'])
                        map_df[f'Modelled_{name}_km']=np.nan
                        if ok.any():
                            map_df.loc[ok,f'Modelled_{name}_km']=bundle['model'].predict(map_df.loc[ok,bundle['features']])
                        modelled_cols.append(f'Modelled_{name}_km')
                    color_options = modelled_cols + color_options
                    if color_options:
                        map_c1,map_c2,map_c3,map_c4=st.columns([1.1,1,0.9,0.6])
                        _crust1_grid   = read_default_crust_grid(str(DEFAULT_CRUST1_GRID))
                        _lithoref_grid = read_default_lithoref18()
                        _bg_choices = ['None']
                        if not _crust1_grid.empty:   _bg_choices.insert(0,'CRUST1.0')
                        if not _lithoref_grid.empty: _bg_choices.insert(min(1,len(_bg_choices)),'LithoRef18 / Alfonso 2019')
                        bg_grid_choice = map_c1.selectbox('Background grid', _bg_choices, index=0, key='training_map_bg_grid')
                        bg_display_mode = map_c1.radio('Display as', ['Surface','Points'], horizontal=True, key='training_map_bg_mode',
                                                       help='Surface resamples to a uniform 1°×1° layer. Points shows raw grid nodes.')
                        normalize_widget_state('training_model_map_layer',color_options)
                        color_by=map_c2.selectbox('Sample layer',color_options,index=0,key='training_model_map_layer',format_func=lambda c: model_map_option_label(c,len(modelled_cols)))
                        zoom_to_data=map_c4.checkbox('Zoom to data',value=False,key='training_map_zoom_to_data')
                        m=plot_df(map_df.dropna(subset=['Lat','Lon']))
                        color_by=resolve_option(color_by,list(m.columns),0)
                        m_geo = geo_downsample(m)
                        if len(m_geo) < len(m): map_c3.caption(f'{len(m_geo):,} / {len(m):,} pts')
                        hover_columns=hover_cols(m_geo,['Sample_ID','Age_Ma','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type_Model','Crust_Thickness'],color_by,'Lat','Lon')
                        fig=px.scatter_geo(
                            m_geo,lat='Lat',lon='Lon',color=color_by,
                            hover_data=map_hover_data(m_geo,hover_columns),
                            projection='natural earth',template='plotly_white',
                            labels={color_by:model_map_legend_title(color_by).replace('<br>',' ')},
                            **map_color_kw(color_by, m_geo),
                        )
                        fig.update_traces(marker=dict(line=dict(color='black',width=0.7)),selector=dict(type='scattergeo'))
                        if bg_grid_choice == 'CRUST1.0' and not _crust1_grid.empty:
                            fig = add_grid_background_to_geofig(fig, _crust1_grid, 'CRUST1_Total_Crust_km', 'CRUST1.0', 'CRUST1.0', bg_display_mode)
                        elif bg_grid_choice == 'LithoRef18 / Alfonso 2019' and not _lithoref_grid.empty:
                            fig = add_grid_background_to_geofig(fig, _lithoref_grid, 'LithoRef18_Total_Crust_km', 'LithoRef18 / Alfonso 2019', 'LithoRef18', bg_display_mode)
                        fig=style_training_map_legends(fig,color_by)
                        fig=apply_geo_zoom_to_data(fig,m,zoom_to_data)
                        fig.update_layout(height=650)
                        st.plotly_chart(fig,width='stretch')

                        # ── Coverage density heat-map ─────────────────────────────────
                        if map_c3.checkbox('Show coverage density', value=False, key='training_map_density'):
                            _lats = pd.to_numeric(m['Lat'], errors='coerce').dropna()
                            _lons = pd.to_numeric(m['Lon'], errors='coerce').dropna()
                            if len(_lats) >= 10:
                                _grid_deg = 10  # 10° × 10° cells
                                _lat_bins = np.arange(-90, 91, _grid_deg)
                                _lon_bins = np.arange(-180, 181, _grid_deg)
                                _hist, _, _ = np.histogram2d(_lats, _lons, bins=[_lat_bins, _lon_bins])
                                _cell_lats = (_lat_bins[:-1] + _lat_bins[1:]) / 2
                                _cell_lons = (_lon_bins[:-1] + _lon_bins[1:]) / 2
                                _clon, _clat = np.meshgrid(_cell_lons, _cell_lats)
                                _dens_df = pd.DataFrame({
                                    'lat': _clat.ravel(), 'lon': _clon.ravel(), 'n': _hist.ravel()
                                })
                                _dens_df = _dens_df[_dens_df['n'] == 0].copy()
                                if not _dens_df.empty:
                                    st.caption(f'**{len(_dens_df)} empty {_grid_deg}° cells** (no training samples) — predictions into these regions are extrapolations.')
                                    _gap_fig = px.scatter_geo(
                                        _dens_df, lat='lat', lon='lon',
                                        projection='natural earth', template='plotly_white',
                                        color_discrete_sequence=['#ef4444'],
                                    )
                                    _gap_fig.update_traces(marker=dict(size=8, symbol='square', opacity=0.4,
                                                                       line=dict(color='#b91c1c', width=0.5)))
                                    _gap_fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0),
                                                           showlegend=False)
                                    st.plotly_chart(_gap_fig, width='stretch', key='training_coverage_gaps')
                                else:
                                    st.success(f'All {_grid_deg}° cells have at least one training sample.')
                    else:
                        st.info('No colour columns found for the training map.')
                else:
                    st.info('Training map needs Lat and Lon columns.')

        # ── Per-group model assignment (Model tab) ───────────────────────────
        if models and 'primary_train_df' in locals() and not primary_train_df.empty and 'Group_ID' in primary_train_df.columns:
            with st.expander('🔬 Per-group model assignment', expanded=True):
                st.caption('This training dataset has group labels — assign a different trained model to each group and run predictions on grouped data.')
                per_group_model_panel(primary_train_df, 'model', models, la_mode)

with t_validation:
    st.header('Validation')
    st.caption('Use this tab to test trained models against known crustal-thickness data that were not used for training.')
    _used_train_sources=st.session_state.get('_training_sources_used',set())
    # Reference datasets (Guo / Zou / Luffi / custom) appear directly. Each
    # reference is filtered out if the user has already used it for training
    # — validating against the same data is meaningless.
    _builtin_val = [
        r for r in available_reference_labels()
        if r not in _used_train_sources
    ]
    # User datasets: only those assigned 'Validate' in the Prepare tab.
    # Hide pool entries whose stem matches a built-in shown above —
    # selecting the friendly name will pick up the prepared version
    # automatically via read_training_source.
    _builtin_val_stems = {
        _BUILTIN_TO_FILE_STEM[_b] for _b in _builtin_val if _b in _BUILTIN_TO_FILE_STEM
    }
    _val_pool_opts = [
        n for n in get_pool_for_tab('Validate')
        if n not in _builtin_val and n not in _builtin_val_stems
    ]
    _val_src_opts = _builtin_val + _val_pool_opts
    if not _val_src_opts:
        st.info('No validation dataset available. Either click a reference '
                'button (Guo / Zou / Luffi / custom refs) in the **Prepare** '
                'tab\'s reference loader, or upload a file there and assign '
                'it **Validate**.')
        test_df = pd.DataFrame()
    else:
        _val_src=st.selectbox('Validation dataset',_val_src_opts,index=0,key='val_data_source',
                              help='References load directly; pool entries from Prepare appear when assigned Validate.')
        # Unified loader handles built-in references (with prepared-pool
        # preference), pool entries by name, and custom refs from the registry.
        test_df = load_dataset_by_name(_val_src)
        if test_df.empty:
            st.warning(f"'{_val_src}' could not be loaded.")
        else:
            st.success(f"Using **{_val_src}**: {len(test_df):,} rows.")
    if not test_df.empty:
        _val_pmt = st.session_state.get('primary_model_type', 'Machine Learning')
        _val_proxy_col = st.session_state.get('_primary_proxy_col', '')
        _val_proxy_name = st.session_state.get('_primary_proxy_name', _val_proxy_col)
        _val_ratio_mode = (_val_pmt != 'Machine Learning' and bool(_val_proxy_col))
        if validation_df.empty and not _val_ratio_mode:
            st.info('Train at least one model in the Model tab, or switch to **Multi-ratio** / **Single-ratio** mode to benchmark without an ML model.')
        else:
            test_df,test_target=target_column_controls('Validation dataset',test_df)
            if test_target is None:
                st.warning('Choose a validation target column before running validation.')
            else:
                test_df=training_subset_controls('Validation dataset',test_df,expanded=True,target=test_target,noun='validation')
                # Granularity toggle — only renders when test_df has Group_ID.
                test_df, _val_granularity = dataset_granularity_controls(
                    test_df, 'val_test', label='Validation data',
                )
                # Single/Multi model toggle — only renders when >1 trained model
                # exists.  In single mode, only the chosen model is benchmarked
                # and the per-group assignment panel is hidden.
                _val_active_models, _val_model_mode = model_selection_controls(
                    models, 'val',
                )
                # Peek at the cached bench so we can decide whether to offer
                # the "Group rows by" picker / show the per-group filter.
                # The bench is read again below for plotting; reading here
                # is cheap (it's a session-state dict lookup).
                _val_cached_bench = st.session_state.get('_val_done_bench', pd.DataFrame())

                # Discover every column that could plausibly be used as a
                # grouping axis: object/category dtypes plus low-cardinality
                # numerics that look like labels (e.g. ``Group_ID`` integers).
                # Sample_ID is excluded because grouping by sample is
                # nonsense — every group would have N=1.
                def _val_group_col_candidates(df):
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        return []
                    cands = []
                    _excl = {'Sample_ID', 'Sample_Name', 'Sample', 'Lat', 'Lon',
                             'Latitude', 'Longitude', 'Observed_km', 'Predicted_km',
                             'Residual_km', 'Known_Thickness_km'}
                    for c in df.columns:
                        if c in _excl or str(c).startswith('_'):
                            continue
                        s = df[c].dropna()
                        if s.empty:
                            continue
                        n_unique = s.nunique()
                        # Need ≥ 2 distinct values for it to be a useful split,
                        # and ≤ ~200 to remain a categorical (otherwise
                        # everything's its own group).
                        if 2 <= n_unique <= 300:
                            # Skip numeric-typed columns that aren't naturally
                            # categorical (only let int-like through).
                            if pd.api.types.is_numeric_dtype(s):
                                if not (c.endswith('_ID') or c == 'Group_ID'
                                        or n_unique <= 30):
                                    continue
                            cands.append(c)
                    # ``Arc_or_Segment`` is a fallback union built by
                    # enrich() (Arc when present, Segment otherwise). When
                    # the bench actually has Arc AND Segment, the union
                    # adds no new information — drop it from the picker
                    # so users don't pick a confusing third axis. Keep it
                    # available when only the union exists (rare edge case).
                    if 'Arc' in cands and 'Segment' in cands and 'Arc_or_Segment' in cands:
                        cands = [c for c in cands if c != 'Arc_or_Segment']
                    # Stable order: Group_Name / Group_ID first, then Arc,
                    # Segment, classic regional/tectonic columns, then alpha.
                    # Geologic_* labels share a coarse→fine hierarchy
                    # (Era ⊃ Period ⊃ Epoch ⊃ Age_Label) so they sit
                    # together right after Tectonic_Setting / Geologic_Domain.
                    _priority = ['Group_Name', 'Group_ID', 'Arc', 'Segment',
                                 'Tectonic_Setting', 'Geologic_Domain',
                                 'Geologic_Era', 'Geologic_Period',
                                 'Geologic_Epoch', 'Geologic_Age_Label',
                                 'Rock_Type_Model', 'Dataset', 'Country',
                                 'Province', 'Terrane', 'Formation', 'Model',
                                 'Arc_or_Segment']
                    out = [c for c in _priority if c in cands]
                    out += [c for c in cands if c not in out]
                    return out

                _val_group_candidates = _val_group_col_candidates(_val_cached_bench)
                _val_has_groups = bool(_val_group_candidates)

                # "Group rows by" picker (primary) + optional "and also by"
                # composite picker (secondary). Composite supports either a
                # second categorical column (Geologic_Period, Tectonic_Setting,
                # …) or numeric Age bins of an arbitrary width — covering the
                # space × time grouping real geology demands.
                _val_group_col = None
                _val_effective_group_col = None  # resolved composite column name
                _plot_bench_grouped = _val_cached_bench  # bench with composite col attached
                if _val_has_groups:
                    _val_group_default = next(
                        (c for c in ('Group_Name', 'Group_ID') if c in _val_group_candidates),
                        _val_group_candidates[0],
                    )
                    _gp1, _gp2, _gp3 = st.columns([1.4, 1.4, 0.8])
                    _val_group_col = _gp1.selectbox(
                        'Group rows by',
                        _val_group_candidates,
                        index=_val_group_candidates.index(
                            st.session_state.get('_val_group_col', _val_group_default)
                            if st.session_state.get('_val_group_col', _val_group_default) in _val_group_candidates
                            else _val_group_default
                        ),
                        key='_val_group_col',
                        format_func=group_picker_label,
                        help='Primary grouping axis. `Group_Name` is the column '
                             'applied in the Group tab; the others (Arc, Segment, '
                             'Tectonic_Setting, Geologic_Period …) come straight '
                             'from the dataset. The Geologic_* columns share a '
                             'coarse→fine hierarchy (Era ⊃ Period ⊃ Epoch ⊃ '
                             'Age_Label). Drives the Show-groups filter, '
                             'the per-group averages table, and the per-group '
                             'overlay on the chart.',
                    )
                    # Secondary composite — none / age-bin / another categorical.
                    # The age-bin path is the headline feature here: lets you
                    # see e.g. "Arc × 10-Ma slices" without re-running the
                    # Group tab.
                    _val_secondary_choices = ['(none)']
                    if 'Age_Ma' in _val_cached_bench.columns and _val_cached_bench['Age_Ma'].notna().any():
                        _val_secondary_choices.append('Age bins (Ma)')
                    _val_secondary_choices += [c for c in _val_group_candidates if c != _val_group_col]
                    _val_secondary_choice = _gp2.selectbox(
                        'And also by (composite)',
                        _val_secondary_choices,
                        index=0, key='_val_group_col_secondary',
                        format_func=group_picker_label,
                        help='Optional second axis. When set, groups are formed '
                             'by combining the primary and secondary into one '
                             'composite ID, e.g. "Andes · Cretaceous" or '
                             '"Andes · 0–10 Ma". Pick "Age bins (Ma)" to slice '
                             'numeric Age_Ma at the chosen width — perfect for '
                             'GAME-style space × time crustal-thickness '
                             'reconstructions.',
                    )
                    _val_age_bin_ma = None
                    if _val_secondary_choice == 'Age bins (Ma)':
                        _val_age_bin_ma = _gp3.slider(
                            'Bin width (Ma)', 1, 200, 10, 1,
                            key='_val_age_bin_ma',
                            help='Width of each Age_Ma bin. Smaller = finer time '
                                 'resolution but smaller groups; 10 Ma is a sensible '
                                 'default for arc geochronology.',
                        )
                    # Resolve composite into an effective grouping column on a
                    # local copy of the bench (so the synthetic ``_Composite_Group``
                    # column doesn't leak back into the cached state).
                    _val_secondary_col_arg = (None if _val_secondary_choice in ('(none)', 'Age bins (Ma)')
                                               else _val_secondary_choice)
                    _plot_bench_grouped, _val_effective_group_col = composite_group_column(
                        _val_cached_bench, _val_group_col,
                        secondary_col=_val_secondary_col_arg,
                        age_bin_ma=(int(_val_age_bin_ma) if _val_age_bin_ma else None),
                        dest_col='_Composite_Group',
                    )
                    if _val_effective_group_col != _val_group_col:
                        _gp3.caption(f'**{int(_plot_bench_grouped[_val_effective_group_col].nunique())}** composite groups')
                    # The downstream code below was written against
                    # ``_val_group_col`` — repoint it at the effective column
                    # so Show-groups, per-group averages and the overlay all
                    # honour the composite when one is picked.
                    _val_group_col = _val_effective_group_col

                # Per-group visibility — only when a grouping column is
                # picked above. Defaults to "all groups visible"; unchecking
                # a group hides its points (and best-fit / moving-average
                # traces, since those are computed only on the visible
                # subset). Independent of the colour-by selector — you can
                # colour by Model and still filter to one group, or vice versa.
                _val_visible_groups = None
                if _val_group_col:
                    # Read the unique groups from the composite-augmented bench
                    # (so when the user picks a composite, the multiselect
                    # shows the synthetic "Andes · 0–10 Ma" labels).
                    _val_all_groups = sorted(
                        str(g) for g in _plot_bench_grouped[_val_group_col].dropna().astype(str).unique()
                    )
                    _gv1, _gv2 = st.columns([4, 1])
                    _val_visible_groups = _gv1.multiselect(
                        f'Show groups (`{_val_group_col}`)',
                        _val_all_groups,
                        default=st.session_state.get(
                            f'_val_visible_groups::{_val_group_col}',
                            _val_all_groups,
                        ),
                        key=f'_val_visible_groups::{_val_group_col}',
                        help='Untick a group to hide its points (and best-fit / '
                             'moving-average lines for that group). Cleared '
                             'rows are still in the cached bench — toggling '
                             'them back does not require re-running.',
                    )
                    if _gv2.button('Reset', key=f'_val_visible_groups_reset::{_val_group_col}',
                                   help='Re-show every group'):
                        st.session_state[f'_val_visible_groups::{_val_group_col}'] = _val_all_groups
                        st.rerun()
                # ── Validation is now explicit. Build a config signature
                # of everything that affects the result, then only run the
                # heavy benchmarking when the user clicks ▶ Run validation.
                # Until pressed, we read the previous test_bench from
                # session state so the user can browse the existing plots
                # without re-running on every interaction.
                import hashlib as _hl_val
                _val_cfg_payload = json.dumps({
                    'src':          _val_src,
                    'target':       test_target,
                    'rows':         int(len(test_df)),
                    'columns':      list(test_df.columns)[:80],
                    'ratio_mode':   bool(_val_ratio_mode),
                    'proxy_col':    str(_val_proxy_col),
                    'pmt':          str(_val_pmt),
                    'model_mode':   str(_val_model_mode),
                    'active_models': sorted(list(_val_active_models.keys())),
                    'la_mode':      str(la_mode),
                    'seed':         int(seed),
                    'granularity':  _val_granularity,
                }, sort_keys=True, default=str)
                _val_sig = _hl_val.md5(_val_cfg_payload.encode('utf-8')).hexdigest()
                _val_run_btn_c1, _val_run_btn_c2 = st.columns([1, 4])
                _val_done_sig = st.session_state.get('_val_done_sig')
                _val_stale = (_val_done_sig != _val_sig) and (_val_active_models or _val_ratio_mode)

                with _val_run_btn_c1:
                    if st.button('▶ Run validation',
                                 key='val_run_btn',
                                 use_container_width=True,
                                 type='primary' if _val_stale or _val_done_sig is None else 'secondary',
                                 help='Run benchmark on the chosen validation dataset. Cached '
                                      'until the config above changes.'):
                        if _val_ratio_mode:
                            _need_game_val = (_val_proxy_col == 'H_GAME_LuffiDucea2022_km')
                            _val_enriched = enrich(test_df.copy(), la_mode, include_game=_need_game_val)
                            if _val_proxy_col not in _val_enriched.columns or _val_enriched[_val_proxy_col].isna().all():
                                st.warning(f'Could not compute **{_val_proxy_name}** from this dataset — '
                                           'check that the required element columns are present.')
                                _new_test_bench = pd.DataFrame()
                            else:
                                _val_target_vals = pd.to_numeric(_val_enriched[test_target], errors='coerce')
                                _val_proxy_vals  = pd.to_numeric(_val_enriched[_val_proxy_col], errors='coerce')
                                _val_ok = _val_target_vals.notna() & _val_proxy_vals.notna()
                                _val_sub = _val_enriched[_val_ok].copy()
                                _new_test_bench = _val_sub.copy()
                                _new_test_bench['Observed_km']  = pd.to_numeric(_val_sub[test_target], errors='coerce')
                                _new_test_bench['Known_Thickness_km'] = _new_test_bench['Observed_km']
                                _new_test_bench['Predicted_km'] = pd.to_numeric(_val_sub[_val_proxy_col], errors='coerce')
                                _new_test_bench['Residual_km']  = _new_test_bench['Predicted_km'] - _new_test_bench['Observed_km']
                                _new_test_bench['Model']     = _val_proxy_name
                                _new_test_bench['Algorithm'] = _val_pmt
                                _new_test_bench = ensure_unique_columns(_new_test_bench.reset_index(drop=True))
                        else:
                            _new_test_bench = benchmark_uploaded(_val_active_models, test_df, test_target, seed, la_mode)
                        if not _new_test_bench.empty and {'Lat','Lon'}.issubset(_new_test_bench):
                            _new_test_bench = attach_crust1_reference(_new_test_bench)
                        st.session_state['_val_done_bench']  = _new_test_bench
                        st.session_state['_val_done_sig']    = _val_sig
                        st.session_state['_rs_val_bench']    = _new_test_bench.copy() if not _new_test_bench.empty else pd.DataFrame()
                        st.rerun()
                with _val_run_btn_c2:
                    if _val_done_sig is None:
                        st.caption('Press **▶ Run validation** to benchmark the trained models against this dataset.')
                    elif _val_stale:
                        st.warning(
                            '⚠ Config changed since the last validation run. Press '
                            '**▶ Run validation** to refresh — plots below show the previous run.'
                        )
                    else:
                        _prev_bench = st.session_state.get('_val_done_bench', pd.DataFrame())
                        st.caption(f'✓ Showing the last validation run ({len(_prev_bench):,} row(s)).')

                # Read result from session state (or empty)
                test_bench = st.session_state.get('_val_done_bench', pd.DataFrame())
                if test_bench is None:
                    test_bench = pd.DataFrame()
                if test_bench.empty:
                    st.warning('No complete blind validation rows could be benchmarked for the selected models.')
                    if not _val_ratio_mode:
                        # Run readiness on the *enriched* frame because that's
                        # what benchmark_uploaded sees — otherwise readiness
                        # can say "ready" while the benchmark loop quietly
                        # drops every row (e.g. enrich() recomputes Sr_Y from
                        # missing Sr/Y columns and overwrites the user-supplied
                        # values with NaN). The two frames must agree or the
                        # diagnostic message becomes a lie.
                        try:
                            _val_test_for_diag = enrich(test_df.copy(), la_mode, include_game=True)
                        except Exception:
                            _val_test_for_diag = test_df
                        report = benchmark_readiness_report(_val_active_models, _val_test_for_diag, test_target)
                        if not report.empty:
                            # Build per-feature blocker counts (which features
                            # cause the most zero-row exclusions). Tells the
                            # user *which* columns to either drop, fill, or
                            # bring in via Prepare-tab mapping.
                            _blockers_per_model = []
                            for _m_name, _m_bundle in _val_active_models.items():
                                _need = list(_m_bundle.get('features', [])) + [test_target]
                                _have = [c for c in _need if c in _val_test_for_diag.columns]
                                if not _have:
                                    continue
                                _num = _val_test_for_diag[_have].apply(pd.to_numeric, errors='coerce')
                                _na_per_col = _num.isna().sum().sort_values(ascending=False)
                                _na_per_col = _na_per_col[_na_per_col > 0]
                                for _c, _n in _na_per_col.items():
                                    _blockers_per_model.append({
                                        'Model':   _m_name,
                                        'Feature': _c,
                                        'Rows missing / non-numeric': int(_n),
                                        '% of dataset': round(100.0 * float(_n) / max(1, len(_val_test_for_diag)), 1),
                                    })
                            _blocker_df = (pd.DataFrame(_blockers_per_model)
                                             .sort_values('Rows missing / non-numeric', ascending=False)
                                           if _blockers_per_model else pd.DataFrame())

                            st.error(benchmark_failure_message(report))
                            # If readiness says rows are available but the
                            # cached bench is empty, that's a real
                            # disagreement (not just a stale cache) —
                            # auto-recover by re-running benchmark_uploaded
                            # inline and comparing. If it now returns rows
                            # we silently store them; if it still returns 0
                            # we surface the actual exception (or row count)
                            # so the user has something concrete to act on.
                            if (report['Complete_rows'].fillna(0).sum() > 0):
                                st.info(
                                    'Readiness says rows are available — attempting an inline '
                                    'recovery run of `benchmark_uploaded(...)` against the same '
                                    'frame to diagnose…'
                                )
                                _recovery_err = None
                                _recovery_bench = pd.DataFrame()
                                try:
                                    _recovery_bench = benchmark_uploaded(
                                        _val_active_models, test_df, test_target, seed, la_mode,
                                    )
                                    if (not _recovery_bench.empty
                                            and {'Lat','Lon'}.issubset(_recovery_bench)):
                                        _recovery_bench = attach_crust1_reference(_recovery_bench)
                                except Exception as _rerr:
                                    _recovery_err = f'{type(_rerr).__name__}: {_rerr}'
                                if _recovery_err is not None:
                                    st.error(
                                        f'`benchmark_uploaded` raised: **{_recovery_err}** — '
                                        'this is the real reason no rows were produced. '
                                        'Capture the traceback in the terminal for the full chain.'
                                    )
                                elif not _recovery_bench.empty:
                                    # The previous empty cache was likely from
                                    # an older config — silently update and
                                    # rerun so the user sees the real result
                                    # without having to press the button again.
                                    st.session_state['_val_done_bench'] = _recovery_bench
                                    st.session_state['_val_done_sig']   = _val_sig
                                    st.session_state['_rs_val_bench']   = _recovery_bench.copy()
                                    st.success(
                                        f'Recovery succeeded — produced **{len(_recovery_bench):,}** '
                                        'benchmark rows. Refreshing…'
                                    )
                                    st.rerun()
                                else:
                                    # Recovery returned empty too — mirror
                                    # benchmark_uploaded's full pipeline
                                    # inline so we can see exactly which step
                                    # collapses to zero. Steps mirror the
                                    # function literally: enrich → column
                                    # presence → numeric coerce → notna mask
                                    # → predict → tmp build.
                                    st.error(
                                        'Recovery also returned 0 rows even though readiness '
                                        f'reports {int(report["Complete_rows"].fillna(0).sum()):,} '
                                        'complete rows. Per-model trace below shows where rows '
                                        'are being dropped:'
                                    )
                                    _trace_rows = []
                                    _enriched = enrich(test_df.copy(), la_mode, include_game=True)
                                    _trace_rows.append({
                                        'Model': '<all>',
                                        'Stage': '0. enrich() output rows',
                                        'Rows surviving': int(len(_enriched)),
                                        'Detail': f'columns: {len(_enriched.columns)}; '
                                                  f'duplicate cols: {int(_enriched.columns.duplicated().sum())}',
                                    })
                                    for _m_name, _m_bundle in _val_active_models.items():
                                        _feats = list(_m_bundle.get('features', []))
                                        _trace_rows.append({
                                            'Model': _m_name,
                                            'Stage': '1. bundle features count',
                                            'Rows surviving': int(len(_enriched)),
                                            'Detail': f'features: {len(_feats)} → {", ".join(_feats[:8])}{"…" if len(_feats) > 8 else ""}',
                                        })
                                        _need = _feats + [test_target]
                                        _missing_in_enriched = [c for c in _need if c not in _enriched.columns]
                                        if _missing_in_enriched:
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '2. column-presence check',
                                                'Rows surviving': 0,
                                                'Detail': f'MISSING: {", ".join(_missing_in_enriched)}',
                                            })
                                            continue
                                        _need_dedup = list(dict.fromkeys(_need))
                                        try:
                                            _num = _enriched[_need_dedup].apply(pd.to_numeric, errors='coerce')
                                            _notna_count = int(_num.notna().all(axis=1).sum())
                                        except Exception as _ne:
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '3. numeric coerce',
                                                'Rows surviving': 0,
                                                'Detail': f'EXCEPTION: {type(_ne).__name__}: {_ne}',
                                            })
                                            continue
                                        _trace_rows.append({
                                            'Model': _m_name,
                                            'Stage': '3. after notna(all-axis=1) on need',
                                            'Rows surviving': _notna_count,
                                            'Detail': 'should match readiness Complete_rows',
                                        })
                                        if _notna_count == 0:
                                            continue
                                        # Step 4 — actually call predict on the
                                        # surviving rows; this is the most
                                        # likely place a silent failure has
                                        # been hiding.
                                        _clean_idx = _num.notna().all(axis=1)
                                        _X_pred = _enriched.loc[_clean_idx, _feats].apply(pd.to_numeric, errors='coerce')
                                        try:
                                            _pred_test = _m_bundle['model'].predict(_X_pred)
                                            _pred_n = int(np.asarray(_pred_test).size)
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '4. model.predict(X)',
                                                'Rows surviving': _pred_n,
                                                'Detail': f'predict succeeded; X shape: {_X_pred.shape}',
                                            })
                                        except Exception as _pe:
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '4. model.predict(X)',
                                                'Rows surviving': 0,
                                                'Detail': f'EXCEPTION: {type(_pe).__name__}: {_pe}',
                                            })
                                            continue
                                        # Step 5 — replicate the tmp assembly
                                        # exactly as benchmark_uploaded does.
                                        # The most likely silent failure here
                                        # is "clean[target] returns DataFrame
                                        # because target column is duplicated
                                        # in clean", which makes .values.ravel()
                                        # produce a 2N-length array → assignment
                                        # raises ValueError → we'd see it here.
                                        try:
                                            _proxy_thickness_cols = list(PROXY_THICKNESS_LABELS) if 'PROXY_THICKNESS_LABELS' in globals() else []
                                            _proxy_value_cols = list(PROXY_VALUE_LIBRARY) if 'PROXY_VALUE_LIBRARY' in globals() else []
                                            _game_cols = ['GAME_Luffi2022_N_mohometers','GAME_Luffi2022_N_raw_mohometers','GAME_Luffi2022_MAD_km','GAME_Luffi2022_IQR_km','GAME_Luffi2022_CI95_Low_km','GAME_Luffi2022_CI95_High_km','GAME_Luffi2022_CI95_Width_km','GAME_Luffi2022_Reliability','GAME_Luffi2022_Status']
                                            _meta_priority_dbg = ['Sample_ID','Lat','Lon','Age_Ma','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type','Rock_Type_Model','MgO','SiO2',test_target]
                                            _meta_cols_dbg = [c for c in _meta_priority_dbg + _proxy_value_cols + _proxy_thickness_cols + _game_cols if c in _enriched]
                                            _meta_set_dbg  = set(_meta_cols_dbg)
                                            _meta_cols_dbg += [c for c in _enriched.columns if c not in _meta_set_dbg]
                                            _clean_dbg = _enriched[list(dict.fromkeys(_need + _meta_cols_dbg))].copy()
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '5a. clean = test_df[need+meta]',
                                                'Rows surviving': int(len(_clean_dbg)),
                                                'Detail': f'cols: {len(_clean_dbg.columns)}; '
                                                          f'duplicate cols: {int(_clean_dbg.columns.duplicated().sum())}',
                                            })
                                            _numeric_need_dbg = _clean_dbg[_need].apply(pd.to_numeric, errors='coerce')
                                            _clean_dbg = _clean_dbg.loc[_numeric_need_dbg.notna().all(axis=1)].copy()
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '5b. clean.loc[notna]',
                                                'Rows surviving': int(len(_clean_dbg)),
                                                'Detail': f'should match step 3',
                                            })
                                            # The risky line — clean[target]
                                            _t_extract = _clean_dbg[test_target]
                                            _t_shape = (_t_extract.shape if hasattr(_t_extract, 'shape') else 'scalar')
                                            _t_kind  = type(_t_extract).__name__
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '5c. clean[target] type',
                                                'Rows surviving': int(len(_clean_dbg)),
                                                'Detail': f'clean[{test_target!r}] is {_t_kind}, shape={_t_shape}',
                                            })
                                            # Try the actual assignment that benchmark_uploaded does
                                            _tmp_dbg = _clean_dbg[list(dict.fromkeys([c for c in _meta_cols_dbg if c != test_target]))].reset_index(drop=True)
                                            _obs_arr = pd.to_numeric(_clean_dbg[test_target], errors='coerce').values.ravel()
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '5d. Observed_km array',
                                                'Rows surviving': int(_obs_arr.size),
                                                'Detail': f'tmp rows: {len(_tmp_dbg)}; obs len: {_obs_arr.size}'
                                                          + (' ⚠ MISMATCH' if _obs_arr.size != len(_tmp_dbg) else ''),
                                            })
                                            if _obs_arr.size == len(_tmp_dbg):
                                                _tmp_dbg['Observed_km']  = _obs_arr
                                                _tmp_dbg['Predicted_km'] = _pred_test
                                                _tmp_dbg['Model']        = _m_name
                                                # Final: pd.concat + ensure_unique_columns
                                                _final_dbg = ensure_unique_columns(pd.concat([_tmp_dbg], ignore_index=True))
                                                _trace_rows.append({
                                                    'Model': _m_name,
                                                    'Stage': '6. final concat+ensure_unique',
                                                    'Rows surviving': int(len(_final_dbg)),
                                                    'Detail': f'cols: {len(_final_dbg.columns)}; '
                                                              f'dup cols: {int(_final_dbg.columns.duplicated().sum())}',
                                                })
                                        except Exception as _se:
                                            import traceback as _tb_dbg
                                            _trace_rows.append({
                                                'Model': _m_name,
                                                'Stage': '5/6. tmp assembly',
                                                'Rows surviving': 0,
                                                'Detail': f'EXCEPTION: {type(_se).__name__}: {_se}',
                                            })
                                    if _trace_rows:
                                        st.dataframe(pd.DataFrame(_trace_rows),
                                                     hide_index=True,
                                                     use_container_width=True)
                                        st.caption(
                                            'If step 4 shows an exception, the model bundle '
                                            'expects features in a different shape than the '
                                            'enriched validation frame supplies — usually a '
                                            'mismatched feature list or a mis-named column.'
                                        )
                            if not _blocker_df.empty:
                                st.markdown(
                                    '**Top features blocking complete rows** — these columns '
                                    'have the most missing/non-numeric values in your validation '
                                    'dataset. Either drop them from the model\'s feature list, '
                                    'or map them in the Prepare tab.'
                                )
                                # Show the worst 12 blockers per model — anything
                                # past that is usually noise.
                                st.dataframe(
                                    _blocker_df.groupby('Model', as_index=False, group_keys=False)
                                               .apply(lambda g: g.head(12))
                                               .reset_index(drop=True),
                                    hide_index=True, use_container_width=True,
                                )
                            st.caption('Benchmark readiness by model')
                            table_action_card('Benchmark readiness by model',report,'benchmark_readiness.csv','validation_benchmark_readiness')

                            # ── Column-name diagnostic ───────────────────────
                            # The most common cause of "missing column" errors
                            # is that the validation file's columns weren't
                            # mapped to canonical names in the Prepare tab.
                            # Show side-by-side what the model expects vs.
                            # what's actually in the validation dataset, so
                            # the user can spot e.g. 'SiO2 (wt%)' vs 'SiO2'.
                            with st.expander('🔍 Column-name diagnostic — what the model expects vs. what your validation file has', expanded=True):
                                _val_cols_set = set(test_df.columns)
                                _diag_rows = []
                                for _m_name, _m_bundle in _val_active_models.items():
                                    _m_feats = list(_m_bundle.get('features', []))
                                    for _feat in _m_feats:
                                        _present = _feat in _val_cols_set
                                        # Look for a "looks-like" candidate when missing
                                        _candidate = ''
                                        if not _present:
                                            _feat_lc = _feat.lower()
                                            for _vc in test_df.columns:
                                                _vc_lc = str(_vc).lower()
                                                if (_vc_lc.startswith(_feat_lc + ' ')
                                                        or _vc_lc.startswith(_feat_lc + '_')
                                                        or _vc_lc.startswith(_feat_lc + '(')
                                                        or _vc_lc == _feat_lc + 't'      # FeO ↔ FeOt
                                                        or _vc_lc == _feat_lc + ' (wt%)'
                                                        or _vc_lc == _feat_lc + ' (ppm)'):
                                                    _candidate = str(_vc)
                                                    break
                                        _diag_rows.append({
                                            'Model':       _m_name,
                                            'Required':    _feat,
                                            'Present':     '✅' if _present else '❌',
                                            'Likely match in your file': _candidate,
                                        })
                                if _diag_rows:
                                    _diag_df = pd.DataFrame(_diag_rows)
                                    _missing_only = _diag_df[_diag_df['Present'] == '❌']
                                    if not _missing_only.empty:
                                        st.markdown(
                                            f'**{len(_missing_only)} required column(s) missing.** '
                                            'If "Likely match in your file" suggests a column name, '
                                            'go back to the **Prepare** tab and confirm the mapping '
                                            'for that column.'
                                        )
                                        st.dataframe(_missing_only, hide_index=True, use_container_width=True)
                                    else:
                                        st.success('All required columns are present — the issue is missing values, not missing column names.')
                                st.caption(
                                    f'**Validation dataset has {len(test_df.columns)} columns**: '
                                    + ', '.join(f'`{c}`' for c in list(test_df.columns)[:60])
                                    + (' …' if len(test_df.columns) > 60 else '')
                                )
                else:
                    st.subheader('Validation diagnostics')
                    summary=validation_summary(test_bench)
                    st.caption('R² shows fit quality. RMSE and MAE are average prediction error in Km. Bias is mean model − known thickness; positive means the model overestimates.')
                    table_action_card('Validation summary',display_validation_summary(test_bench),'validation_summary.csv','validation_summary')

                    # ── Performance by 5 km Observed-thickness bin ──────────
                    # Slices the bench by Observed_km into 5 km intervals and
                    # reports N, mean / median Predicted, mean / median Δ
                    # (Predicted − Observed, signed bias), MAE and RMSE per
                    # bin. Highlights where the model systematically over- or
                    # under-predicts. Bin axis is Observed_km because the
                    # question is "at this actual thickness, how does the
                    # model do?" — useful when residuals are not uniform
                    # across the thickness range.
                    _vb_obs  = pd.to_numeric(test_bench.get('Observed_km'),  errors='coerce')
                    _vb_pred = pd.to_numeric(test_bench.get('Predicted_km'), errors='coerce')
                    _vb_ok   = _vb_obs.notna() & _vb_pred.notna()
                    if _vb_ok.any():
                        _vb_df = pd.DataFrame({
                            'Observed_km':  _vb_obs[_vb_ok].to_numpy(),
                            'Predicted_km': _vb_pred[_vb_ok].to_numpy(),
                        })
                        if 'Model' in test_bench.columns:
                            _vb_df['Model'] = test_bench.loc[_vb_ok, 'Model'].astype(str).to_numpy()
                        _vb_df['Delta_km'] = _vb_df['Predicted_km'] - _vb_df['Observed_km']
                        _vb_lo = max(0, int(np.floor(float(_vb_df['Observed_km'].min()) / 5.0) * 5))
                        _vb_hi = int(np.ceil(float(_vb_df['Observed_km'].max()) / 5.0) * 5)
                        if _vb_hi <= _vb_lo:
                            _vb_hi = _vb_lo + 5
                        _vb_edges = np.arange(_vb_lo, _vb_hi + 5, 5)
                        _vb_df['_bin'] = pd.cut(_vb_df['Observed_km'], bins=_vb_edges,
                                                 right=False, include_lowest=True)
                        # Optional per-model split — only when multi-model is
                        # active; single-model collapses to one row per bin.
                        _vb_split_models = ('Model' in _vb_df.columns
                                            and _vb_df['Model'].nunique() > 1)
                        _vb_group_keys = (['Model', '_bin'] if _vb_split_models else ['_bin'])
                        _vb_rows = []
                        for _gk, _g in _vb_df.groupby(_vb_group_keys, dropna=True, observed=True):
                            if _g.empty:
                                continue
                            _bin_obj = _gk[-1] if _vb_split_models else _gk
                            _r = _g['Delta_km']
                            _row = {
                                'Observed bin [km]':   f'{int(_bin_obj.left)}–{int(_bin_obj.right)}',
                                'N':                    int(len(_g)),
                                'Mean Predicted [km]':  round(float(_g['Predicted_km'].mean()),  2),
                                'Median Predicted [km]':round(float(_g['Predicted_km'].median()),2),
                                'Mean Δ [km]':          round(float(_r.mean()),                 2),
                                'Median Δ [km]':        round(float(_r.median()),               2),
                                'MAE [km]':             round(float(_r.abs().mean()),           2),
                                'RMSE [km]':            round(float(np.sqrt((_r ** 2).mean())), 2),
                            }
                            if _vb_split_models:
                                _row = {'Model': _gk[0], **_row}
                            _vb_rows.append(_row)
                        if _vb_rows:
                            _vb_table = pd.DataFrame(_vb_rows)
                            st.caption(
                                'Performance sliced by **Observed_km** in 5 km intervals. '
                                'Δ is signed (Predicted − Observed) so its sign shows the '
                                'direction of bias inside each band; MAE and RMSE show the '
                                'magnitude. Bins where |Δ| or RMSE spike are where the '
                                'model is struggling.'
                            )
                            table_action_card(
                                'Performance by Observed thickness bin (5 km)',
                                _vb_table,
                                'validation_by_thickness_bin.csv',
                                'val_perf_by_bin',
                            )

                    if not importance_df.empty:
                        with st.expander('Feature importance', expanded=False):
                            _render_feature_importance_panel(
                                importance_df, list(models.keys()),
                                'val_importance_model', 'val_importance_sort')
                    # Apply per-group visibility filter (set above when groups
                    # exist on the bench). Also filters the summary recompute
                    # so R²/RMSE/MAE in the table reflect what's plotted.
                    # Uses _val_group_col which is the user-picked grouping
                    # column (Group_Name / Arc / Segment / composite of two,
                    # …). _plot_bench_grouped already carries the synthetic
                    # composite column when one was built, so use it as the
                    # plotting source — falls back to test_bench when no
                    # grouping is in play.
                    _plot_bench = (_plot_bench_grouped
                                    if (_val_group_col is not None
                                        and isinstance(_plot_bench_grouped, pd.DataFrame)
                                        and not _plot_bench_grouped.empty)
                                    else test_bench)
                    if (_val_visible_groups is not None
                            and _val_group_col in _plot_bench.columns
                            and len(_val_visible_groups) < len(_val_all_groups)):
                        _gn_str = _plot_bench[_val_group_col].astype(str)
                        _plot_bench = _plot_bench[_gn_str.isin(_val_visible_groups)].copy()
                        st.caption(
                            f'Showing **{len(_val_visible_groups)} of {len(_val_all_groups)}** '
                            f'`{_val_group_col}` group(s) '
                            f'({len(_plot_bench):,} of {len(test_bench):,} rows). '
                            'Untick groups in **Show groups** above to filter further.'
                        )

                    # ── Per-group averages (single-model focus) ────────────
                    # When the bench carries Group_Name labels, give the
                    # user a per-group breakdown of how the model fares
                    # against the known thickness. One row per group, with
                    # median / mean predicted, observed, signed residual
                    # (model − known), absolute residual, and the spread
                    # (IQR + std). Honours the Show-groups filter.
                    if (_val_group_col
                            and _val_group_col in _plot_bench.columns
                            and _plot_bench[_val_group_col].notna().any()):
                        _grp_rows = []
                        # Only iterate over models actually present in the
                        # filtered bench — single-model mode collapses to one
                        # group of rows here, multi-model splits per model.
                        _models_in_view = (_plot_bench['Model'].dropna().astype(str).unique().tolist()
                                            if 'Model' in _plot_bench.columns else ['(model)'])
                        for _mn in _models_in_view:
                            _msub = (_plot_bench[_plot_bench['Model'].astype(str) == _mn]
                                     if 'Model' in _plot_bench.columns else _plot_bench)
                            for _gname, _g in _msub.groupby(_val_group_col, dropna=True):
                                _obs  = pd.to_numeric(_g.get('Observed_km'),  errors='coerce')
                                _pred = pd.to_numeric(_g.get('Predicted_km'), errors='coerce')
                                _ok = _obs.notna() & _pred.notna()
                                if not _ok.any():
                                    continue
                                _o = _obs[_ok]; _p = _pred[_ok]
                                _resid = _p - _o
                                _grp_rows.append({
                                    'Model':              _mn,
                                    _val_group_col:       str(_gname),
                                    'N':                  int(_ok.sum()),
                                    'Median Observed [km]':  float(_o.median()),
                                    'Median Predicted [km]': float(_p.median()),
                                    'Mean Observed [km]':    float(_o.mean()),
                                    'Mean Predicted [km]':   float(_p.mean()),
                                    'Median Residual [km]':  float(_resid.median()),
                                    'Mean Residual [km]':    float(_resid.mean()),
                                    'MAE [km]':              float(_resid.abs().mean()),
                                    'RMSE [km]':             float(np.sqrt((_resid ** 2).mean())),
                                    'Predicted IQR [km]':    float(_p.quantile(0.75) - _p.quantile(0.25)),
                                    'Predicted SD [km]':     float(_p.std()),
                                })
                        if _grp_rows:
                            _grp_df = pd.DataFrame(_grp_rows).round(2)
                            # Hide the Model column when only one model is in view
                            _show_cols = [c for c in _grp_df.columns
                                          if not (c == 'Model' and _grp_df['Model'].nunique() <= 1)]
                            table_action_card(
                                f'Per-`{_val_group_col}` averages ({len(_grp_df)} group(s))',
                                _grp_df[_show_cols],
                                f'validation_{_val_group_col}_averages.csv',
                                f'val_group_averages_{_val_group_col}',
                            )
                    # ── Validation plot ─────────────────────────────────────
                    # One-stop scatter that replaces the legacy "Proxy comparison"
                    # block + the standalone proxy-thickness chart. X / Y / Colour
                    # are picked from a flat list grouped by category in the
                    # format_func — Crustal thickness estimates (Observed,
                    # Predicted, every H_*_km proxy) and Ratios & values (Sr/Y,
                    # La/Yb_N, SiO2, MgO, etc.). Carries the full proxy-plot
                    # toolkit: formula curves, moving median/mean, outlier
                    # clipping, best-fit, 1:1 line.
                    with st.expander('Validation plot', expanded=True):
                        _um_axis_opts = simple_xy_axis_options(_plot_bench)
                        if not _um_axis_opts:
                            st.info('No usable columns on this bench yet.')
                        else:
                            # What does Predicted_km currently resolve to?
                            # In ML mode it's the active-model prediction; in
                            # ratio / GAME mode it's the chosen proxy. Tell
                            # the user, since the picker categorisation now
                            # honours this.
                            _pmt_now = st.session_state.get('primary_model_type', 'Machine Learning')
                            if _pmt_now == 'Multi-ratio':
                                _pm_proxy = st.session_state.get('_primary_proxy_name', 'GAME consensus')
                                st.caption(f'ℹ️ `Predicted_km` currently = **{_pm_proxy}** (proxy estimate, not an ML prediction). Pick a `Predicted_<model>_km` column to plot a specific trained ML model.')
                            elif _pmt_now == 'Single-ratio':
                                _pm_proxy = st.session_state.get('_primary_proxy_name', 'Single-ratio proxy')
                                st.caption(f'ℹ️ `Predicted_km` currently = **{_pm_proxy}**. Pick a `Predicted_<model>_km` column to plot a specific trained ML model.')
                            else:
                                _pm_primary = st.session_state.get('active_interp_model', '')
                                if _pm_primary:
                                    st.caption(f'ℹ️ `Predicted_km` = **{_pm_primary}** (primary model). Per-model columns `Predicted_<model>_km` let you pick a specific trained model on either axis.')
                            # Default Y: primary-model per-model column if it
                            # exists on the bench, else generic Predicted_km.
                            _pm_safe = _sanitise_model_name(st.session_state.get('active_interp_model', '')) if st.session_state.get('active_interp_model') else ''
                            _pm_y_col = f'Predicted_{_pm_safe}_km' if _pm_safe else ''
                            _pm_resid_col = f'Residual_{_pm_safe}_km' if _pm_safe else ''
                            # Defaults: classic predicted-vs-known scatter
                            # coloured by signed residual (red-cream-blue).
                            _um_x_default = 'Observed_km'  if 'Observed_km'  in _um_axis_opts else _um_axis_opts[0]
                            _um_y_default = (
                                _pm_y_col if _pm_y_col in _um_axis_opts else
                                ('Predicted_km' if 'Predicted_km' in _um_axis_opts else
                                 (_um_axis_opts[1] if len(_um_axis_opts) > 1 else _um_axis_opts[0]))
                            )
                            _um_c_default = (
                                _pm_resid_col if _pm_resid_col in _um_axis_opts else
                                ('Residual_km' if 'Residual_km' in _um_axis_opts else None)
                            )

                            # Two-stage pickers (group → column) for each
                            # axis. Each picker lives in its own column
                            # so the three pairs sit on a single row;
                            # point-size lives in the rightmost column.
                            _ux_x, _ux_y, _ux_c, _ux_sz = st.columns([1.6, 1.6, 1.6, 0.9])
                            _um_x = simple_xy_two_stage_picker(
                                'X axis', _um_axis_opts, _um_x_default,
                                key_prefix='val_xy_x', container=_ux_x,
                            )
                            _um_y = simple_xy_two_stage_picker(
                                'Y axis', _um_axis_opts, _um_y_default,
                                key_prefix='val_xy_y', container=_ux_y,
                            )
                            _um_c = simple_xy_two_stage_picker(
                                'Colour by', _um_axis_opts, _um_c_default,
                                key_prefix='val_xy_c', container=_ux_c,
                                include_none=True,
                            )
                            _ux_sz.markdown('**Point size**')
                            _um_pt_size = _ux_sz.slider('Point size', 2, 12, 6, 1,
                                                         key='val_xy_size',
                                                         label_visibility='collapsed')

                            # Overlays row — show points + reference overlays
                            _uc0, _uc1, _uc2, _uc3 = st.columns(4)
                            _um_show_pts   = _uc0.checkbox('Show points', True, key='val_xy_show_pts',
                                                           help='Hide the sample scatter and keep only the overlays (medians, means, moving summary). The trace stays in the legend so you can flip it back on with one click.')
                            _um_one_to_one = _uc1.checkbox('1:1 line', True,  key='val_xy_121',
                                                           help='Drawn when both axes are thickness columns')
                            _um_curve      = _uc2.checkbox('Formula curve', True, key='val_xy_curve',
                                                           help='Overlay the proxy calibration when (X, Y) matches a known formula pair (e.g. X=Sr_Y, Y=Profeta Sr/Y)')
                            _um_bestfit    = _uc3.checkbox('Best fit', False, key='val_xy_bf')

                            # Per-group statistic overlay — shown only when
                            # the bench has Group_Name labels. Renders one
                            # diamond / star per group at the (median, median)
                            # or (mean, mean) point.
                            _um_has_groups = (_val_group_col is not None
                                              and _val_group_col in _plot_bench.columns
                                              and _plot_bench[_val_group_col].notna().any())
                            _um_group_stat = 'Off'
                            if _um_has_groups:
                                _ug1, _ug2 = st.columns([1.0, 3.0])
                                _um_group_stat = _ug1.selectbox(
                                    'Per-group overlay',
                                    ['Off', 'Median', 'Mean', 'Both'],
                                    index=0, key='val_xy_group_stat',
                                    help=f'Plot one diamond (median) or star (mean) per `{_val_group_col}` group at '
                                         'the centre of that group\'s scatter — useful for spotting '
                                         'systematic offsets between groups when the sample dots '
                                         'are turned off.',
                                )
                                _ug2.caption(
                                    f'Diamonds = group medians · stars = group means. '
                                    f'One marker per `{_val_group_col}` group, coloured from the Bold palette.'
                                )

                            # Moving-summary + clip row — full feature parity
                            # with the legacy proxy plot.
                            _ut1, _ut2, _ut3, _ut4 = st.columns([1.2, 1.0, 1.0, 1.0])
                            _um_trend     = _ut1.selectbox(
                                'Moving summary', ['Off', 'Median', 'Mean', 'Both'],
                                index=0, key='val_xy_trend',
                                help='Bin the X axis at the chosen width and draw '
                                     'the median / mean of Y per bin. Skips bins '
                                     'with fewer than the minimum N points.',
                            )
                            _um_trend_bin = _ut2.slider('Bin width', 0.5, 20.0, 5.0, 0.5,
                                                        key='val_xy_trend_bin',
                                                        help='Width of each X bin (in X-axis units)')
                            _um_trend_n   = _ut3.slider('Min N',     1, 50,   5,  1,
                                                        key='val_xy_trend_min_n',
                                                        help='Skip bins that contain fewer than this many points')
                            _clip_opts    = {'None': 0.0, '0.5%': 0.5, '1%': 1.0, '2%': 2.0, '5%': 5.0}
                            _um_clip      = _clip_opts[_ut4.selectbox(
                                'Clip outliers', list(_clip_opts), index=0, key='val_xy_clip',
                                help='Trim the top / bottom N % of values on both axes before plotting.',
                            )]

                            # GAME card — show only when GAME column lives on this bench
                            if 'H_GAME_LuffiDucea2022_km' in _plot_bench.columns:
                                render_game_settings_card(key_suffix='_val')

                            # Bail early if either axis is empty (e.g. user
                            # picked "(none)" on Colour and that's resolved
                            # to None up-stream).
                            if _um_x is None or _um_y is None:
                                st.info('Pick both an X and a Y column to render the chart.')
                                _um_fig = None
                            else:
                                _um_fig = simple_xy_figure(
                                    _plot_bench, _um_x, _um_y, _um_c,
                                    show_one_to_one=_um_one_to_one,
                                    show_curve=_um_curve,
                                    show_best_fit=_um_bestfit,
                                    point_size=int(_um_pt_size),
                                    height=540,
                                    trend_stat=_um_trend,
                                    trend_bin_width=float(_um_trend_bin),
                                    trend_min_n=int(_um_trend_n),
                                    clip_pct=float(_um_clip),
                                    show_points=bool(_um_show_pts),
                                    group_stat=str(_um_group_stat),
                                    group_col=(_val_group_col or 'Group_Name'),
                                )
                            if _um_fig is None:
                                st.warning('Pick columns that have at least one row of overlapping numeric data.')
                            else:
                                st.plotly_chart(_um_fig, width='stretch')
                                # Print the proxy formula text underneath when the
                                # user has picked a (X, Y) pair we know about.
                                _um_formula_text = PROXY_THICKNESS_FORMULAS.get(_um_y) \
                                                   or PROXY_THICKNESS_FORMULAS.get(_um_x)
                                if _um_curve and _um_formula_text:
                                    st.caption(f'**Formula in view:** {_um_formula_text}')

                    with st.expander('Sample size adequacy', expanded=False):
                        st.caption(
                            'How many samples from your target area do you need for the median prediction to be stable? '
                            'The chart bootstraps the median at each n using this validation dataset and shows the '
                            'confidence interval narrowing as n grows. '
                            'Green and orange bands show ±5 km and ±10 km precision targets — adjust below.'
                        )
                        _ss_c1, _ss_c2, _ss_c3, _ss_c4 = st.columns(4)
                        _ss_model_opts = test_bench['Model'].dropna().astype(str).unique().tolist() if 'Model' in test_bench else model_options
                        _ss_model = _ss_c1.selectbox('Model', _ss_model_opts or ['(none)'], index=0, key='val_ss_model')
                        _ss_ci = _ss_c2.selectbox('Confidence interval (%)', [80, 90, 95, 99], index=2, key='val_ss_ci')
                        _ss_t1 = _ss_c3.number_input('Precision target 1 (±km)', 1.0, 30.0, 5.0, 0.5, key='val_ss_t1')
                        _ss_t2 = _ss_c4.number_input('Precision target 2 (±km)', 1.0, 30.0, 10.0, 0.5, key='val_ss_t2')
                        _ss_vals = pd.to_numeric(
                            test_bench.loc[test_bench['Model'].astype(str) == str(_ss_model), 'Predicted_km'],
                            errors='coerce'
                        ).dropna().values
                        render_sample_size_panel(_ss_vals, ci_pct=int(_ss_ci),
                                                 precision_targets=(_ss_t1, _ss_t2),
                                                 key_prefix='val_ss')

                    if 'H_GAME_LuffiDucea2022_km' in test_bench:
                        # GAME diagnostics — trimmed to the bits the
                        # Validation plot can't reproduce. The old
                        # "Consensus" tab was just a GAME-vs-known scatter
                        # which the Validation plot now does (Y =
                        # H_GAME_LuffiDucea2022_km, Colour = Residual_km
                        # on the red-cream-blue ramp). What's kept:
                        #   • Reliability — N-mohometer survival tracking
                        #     (N_raw vs N_kept after MAD filtering) and
                        #     the per-N median MAD/IQR/CI95 table.
                        #   • Calibration — the per-sensor LOWESS T2
                        #     calibration surface (Sr/Y, La/Yb, Ce/Y,
                        #     elevation, MgO …) showing the published
                        #     calibration domain and where the user's
                        #     samples sit relative to it.
                        with st.expander('GAME diagnostics',expanded=False):
                            st.caption(
                                'For the GAME-vs-known scatter use the **Validation plot** above '
                                'with `H_GAME_LuffiDucea2022_km` on the Y axis. This expander '
                                'keeps the diagnostics that view can\'t reproduce: per-sensor '
                                'calibration surfaces and N-mohometer survival tracking.'
                            )
                            gtab2,gtab3=st.tabs(['Reliability','Calibration'])
                            with gtab2:
                                st.caption('GAME explicitly tracks how many mohometers survive data-availability, reference-model residual/RMSE, and STD/MAD filtering. Low kept N, high MAD/IQR, wide bootstrap CI, or an "all valid; high spread" status should be treated as lower-confidence interpretation.')
                                game_rel_x=st.selectbox('Reference for reliability residual',['Observed_km','Predicted_km'],index=0,key='game_reliability_x',format_func=lambda c: 'known: crustal thickness [Km]' if c == 'Observed_km' else 'model: crustal thickness [Km]')
                                game_rel_size=int(st.session_state.get('global_point_size', 5))
                                rfig=game_reliability_figure(test_bench,game_rel_x,game_rel_size)
                                if rfig is not None:
                                    st.plotly_chart(rfig,width='stretch',key='val_game_reliability_fig')
                                rel_counts=test_bench['GAME_Luffi2022_Reliability'].fillna('unknown').value_counts().rename_axis('Reliability').reset_index(name='Rows') if 'GAME_Luffi2022_Reliability' in test_bench else pd.DataFrame()
                                if not rel_counts.empty:
                                    table_action_card('GAME reliability counts',rel_counts,'game_reliability_counts.csv','game_reliability_counts')
                                rel_summary=game_reliability_summary(test_bench,game_rel_x)
                                if not rel_summary.empty:
                                    table_action_card('GAME reliability by N',rel_summary,'game_reliability_by_n.csv','game_reliability_by_n')
                                # Status counts moved here from the dropped
                                # Consensus tab — small but useful.
                                status_counts=test_bench['GAME_Luffi2022_Status'].fillna('unknown').value_counts().rename_axis('Status').reset_index(name='Rows') if 'GAME_Luffi2022_Status' in test_bench else pd.DataFrame()
                                if not status_counts.empty:
                                    table_action_card('GAME status counts',status_counts,'game_status_counts.csv','game_status_counts')
                            with gtab3:
                                sensor_options=[s for s,_,_ in GAME_SENSORS if s in game_calibrators()]
                                if sensor_options:
                                    sensor=st.selectbox('Mohometer calibration',sensor_options,index=0,key='game_calibration_sensor',format_func=lambda s: game_sensor_labels().get(s,s))
                                    cfig=game_calibration_figure(test_bench,sensor)
                                    if cfig is not None:
                                        st.plotly_chart(cfig,width='stretch',key='val_game_calibration_fig')
                                table_action_card('GAME mohometer summary',game_sensor_summary(),'game_mohometer_summary.csv','game_mohometer_summary')

                    # Legacy "Proxy comparison" block removed — replaced by
                    # the unified Validation plot above which carries the
                    # same proxy/multi-method scatter, formula curves,
                    # moving summary and outlier clipping in a single
                    # picker-driven view.

                    st.subheader('Blind validation map')
                    _render_blind_validation_map(test_bench)

                    # ── Per-group model assignment ───────────────────────────
                    # Hidden in 'single' mode — there's only one model, so
                    # routing per group makes no sense.
                    if (_val_model_mode == 'multi'
                            and models and not test_bench.empty
                            and 'Group_ID' in test_bench.columns):
                        with st.expander('🔬 Per-group model assignment', expanded=True):
                            st.caption('This dataset has group labels — assign a different trained model to each group and run per-group predictions.')
                            per_group_model_panel(test_bench, 'validate', _val_active_models, la_mode, '_rs_val_bench')

                    st.caption('Download validation results in the Summary tab → Export results.')

                    # ── Outlier flagging ─────────────────────────────────────────────────────
                    if 'Residual_km' in test_bench.columns:
                        with st.expander('High-residual sample flagging', expanded=False):
                            st.caption('Identify samples where the model residual (predicted − known) is unusually large. Use to investigate data quality issues or exclude unreliable samples before retraining.')
                            _olf_c1, _olf_c2 = st.columns([1, 1])
                            _olf_thresh_mode = _olf_c1.radio('Flag threshold', ['σ multiplier', 'Fixed Km'], horizontal=True, key='val_olf_mode')
                            _olf_resid = pd.to_numeric(test_bench['Residual_km'], errors='coerce').dropna()
                            _olf_sigma = float(_olf_resid.std()) if len(_olf_resid) > 1 else 1.0
                            if _olf_thresh_mode == 'σ multiplier':
                                _olf_nsig = _olf_c2.slider('N × σ', 1.0, 4.0, 2.0, 0.25, key='val_olf_nsig')
                                _olf_cutoff = _olf_nsig * _olf_sigma
                            else:
                                _olf_cutoff = _olf_c2.number_input('Threshold [Km]', 1.0, 100.0, 10.0, 1.0, key='val_olf_km')
                            _olf_mask = test_bench['Residual_km'].abs() > _olf_cutoff
                            _olf_flagged = test_bench[_olf_mask].copy()
                            st.metric('Flagged samples', int(_olf_mask.sum()), help=f'|Residual| > {_olf_cutoff:.1f} Km')
                            if not _olf_flagged.empty:
                                _olf_show_cols = [c for c in ['Sample_ID','Lat','Lon','Observed_km','Predicted_km','Residual_km','Dataset','Arc_or_Segment','Tectonic_Setting'] if c in _olf_flagged.columns]
                                st.dataframe(
                                    _olf_flagged[_olf_show_cols].sort_values('Residual_km', key=abs, ascending=False),
                                    hide_index=True, use_container_width=True,
                                )
                                st.download_button(
                                    'Download flagged samples',
                                    _olf_flagged.to_csv(index=False).encode('utf-8'),
                                    'high_residual_samples.csv', 'text/csv',
                                    key='val_olf_dl',
                                )
                                _olf_excl_key = 'val_olf_excluded_ids'
                                if st.button('Exclude flagged samples and retrain', key='val_olf_retrain'):
                                    _excl_ids = set(_olf_flagged['Sample_ID'].astype(str)) if 'Sample_ID' in _olf_flagged else set()
                                    st.session_state[_olf_excl_key] = _excl_ids
                                    st.info(f'{len(_excl_ids)} samples marked for exclusion. Return to the Model tab and retrain — excluded samples will be filtered from the training data stored in session state.')
                                    _cur_train = st.session_state.get('dp_training_df', pd.DataFrame())
                                    if not _cur_train.empty and 'Sample_ID' in _cur_train.columns:
                                        _cur_train_filt = _cur_train[~_cur_train['Sample_ID'].astype(str).isin(_excl_ids)].copy()
                                        st.session_state['dp_training_df'] = _cur_train_filt
                                        st.success(f'Training set reduced from {len(_cur_train):,} → {len(_cur_train_filt):,} rows. Retrain in the Model tab.')

with t_unknown:
    st.header('Prediction Explorer')
    st.caption('Upload geochemistry data without known crustal thickness to get model predictions, GAME estimates, and proxy comparisons.')
    # Reference datasets (built-ins + custom refs) appear directly without
    # needing to be loaded through Prepare. Pool entries assigned 'Predict'
    # in Prepare are listed alongside.  Built-in stems (e.g.
    # 'GuoYang_2023_Model') are filtered out — the friendly name covers them.
    _pred_ref_opts  = available_reference_labels()
    _pred_builtin_stems = {
        _BUILTIN_TO_FILE_STEM[r] for r in _pred_ref_opts
        if r in _BUILTIN_TO_FILE_STEM
    }
    _pred_pool_opts = [
        n for n in get_pool_for_tab('Predict')
        if n not in _pred_ref_opts and n not in _pred_builtin_stems
    ]
    _pred_src_opts = _pred_ref_opts + _pred_pool_opts
    if not _pred_src_opts:
        st.info('No prediction dataset available. Either click a reference '
                'button (Guo / Zou / Luffi / your custom refs) in the '
                '**Prepare** tab\'s reference loader, or upload a file there '
                'and assign it **Predict**.')
        uk_raw = pd.DataFrame()
    else:
        _pred_src=st.selectbox('Prediction dataset',_pred_src_opts,index=0,key='pred_data_source',
                               help='References load directly; pool entries from Prepare appear when assigned Predict.')
        uk_raw = load_dataset_by_name(_pred_src)
        if not uk_raw.empty:
            st.success(f"Using **{_pred_src}**: {len(uk_raw):,} rows.")
    if not uk_raw.empty:
        _uk_pmt = st.session_state.get('primary_model_type', 'Machine Learning')
        _uk_proxy_col = st.session_state.get('_primary_proxy_col', '')
        _uk_proxy_name = st.session_state.get('_primary_proxy_name', _uk_proxy_col)
        _uk_ratio_mode = (_uk_pmt != 'Machine Learning' and bool(_uk_proxy_col))
        if not models and not _uk_ratio_mode:
            st.info('Train at least one model in the Model tab, or switch to **Multi-ratio** / **Single-ratio** in the Model tab to run predictions without an ML model.')
        else:
            uk_raw=training_subset_controls('Unknown dataset',uk_raw,expanded=True,target=None,noun='sample')
            # Granularity toggle — only renders when uk_raw has Group_ID.
            uk_raw, _pred_granularity = dataset_granularity_controls(
                uk_raw, 'predict_uk', label='Prediction data',
            )
            # Single/Multi model toggle — only renders when >1 trained model
            # exists.  In single mode, only the chosen model is used and the
            # per-group assignment panel is hidden.
            _pred_active_models, _pred_model_mode = model_selection_controls(
                models, 'predict',
            )
            # ── Predictions are now explicit. Build a config signature
            # and only run the heavy prediction work when the user clicks
            # ▶ Run predictions. Until pressed, we read the previous
            # pred_bench from session state.
            import hashlib as _hl_pred
            _pred_cfg_payload = json.dumps({
                'src':           _pred_src if '_pred_src' in dir() else '',
                'rows':          int(len(uk_raw)),
                'columns':       list(uk_raw.columns)[:80],
                'ratio_mode':    bool(_uk_ratio_mode),
                'proxy_col':     str(_uk_proxy_col),
                'pmt':           str(_uk_pmt),
                'model_mode':    str(_pred_model_mode),
                'active_models': sorted(list(_pred_active_models.keys())),
                'la_mode':       str(la_mode),
                'seed':          int(seed),
                'granularity':   _pred_granularity,
            }, sort_keys=True, default=str)
            _pred_sig = _hl_pred.md5(_pred_cfg_payload.encode('utf-8')).hexdigest()
            _pred_run_btn_c1, _pred_run_btn_c2 = st.columns([1, 4])
            _pred_done_sig = st.session_state.get('_pred_done_sig')
            _pred_stale = (_pred_done_sig != _pred_sig) and (_pred_active_models or _uk_ratio_mode)

            with _pred_run_btn_c1:
                if st.button('▶ Run predictions',
                             key='pred_run_btn',
                             use_container_width=True,
                             type='primary' if _pred_stale or _pred_done_sig is None else 'secondary',
                             help='Run predictions on the chosen dataset. Cached until the '
                                  'config above changes.'):
                    if _uk_ratio_mode:
                        _need_game_uk = (_uk_proxy_col == 'H_GAME_LuffiDucea2022_km')
                        _uk_enriched = enrich(uk_raw.copy(), la_mode, include_game=_need_game_uk)
                        if _uk_proxy_col not in _uk_enriched.columns or _uk_enriched[_uk_proxy_col].isna().all():
                            st.warning(f'Could not compute **{_uk_proxy_name}** from the uploaded data — '
                                       'ensure the required element columns are mapped in the Prepare tab.')
                            _new_pred_bench = pd.DataFrame()
                        else:
                            _uk_proxy_meta = [c for c in [
                                'Sample_ID','Lat','Lon','Age_Ma','Geologic_Era','Geologic_Period','Geologic_Epoch',
                                'Geologic_Age_Label','Tectonic_Setting','Arc_or_Segment','Geologic_Domain',
                                'Dataset','Rock_Type','Rock_Type_Model','MgO','SiO2','Reliability_Flags',
                            ] + list(PROXY_THICKNESS_LABELS) + list(PROXY_VALUE_LIBRARY) + [
                                'GAME_Luffi2022_N_mohometers','GAME_Luffi2022_N_raw_mohometers',
                                'GAME_Luffi2022_MAD_km','GAME_Luffi2022_IQR_km',
                                'GAME_Luffi2022_CI95_Low_km','GAME_Luffi2022_CI95_High_km',
                                'GAME_Luffi2022_CI95_Width_km','GAME_Luffi2022_Reliability','GAME_Luffi2022_Status',
                            ] if c in _uk_enriched.columns]
                            _uk_proxy_extra = [c for c in _uk_enriched.columns if c not in set(_uk_proxy_meta)]
                            _new_pred_bench = _uk_enriched[list(dict.fromkeys(_uk_proxy_meta + _uk_proxy_extra))].copy()
                            _new_pred_bench['Predicted_km'] = pd.to_numeric(_uk_enriched[_uk_proxy_col], errors='coerce')
                            _new_pred_bench['Model'] = _uk_proxy_name
                            _new_pred_bench['Algorithm'] = _uk_pmt
                            _new_pred_bench = ensure_unique_columns(_new_pred_bench.dropna(subset=['Predicted_km']).reset_index(drop=True))
                    else:
                        _new_pred_bench = predict_uploaded(_pred_active_models, uk_raw, seed, la_mode)
                    if not _new_pred_bench.empty and {'Lat','Lon'}.issubset(_new_pred_bench):
                        _new_pred_bench = attach_crust1_reference(_new_pred_bench)
                    st.session_state['_pred_done_bench'] = _new_pred_bench
                    st.session_state['_pred_done_sig']   = _pred_sig
                    if not _new_pred_bench.empty:
                        st.session_state['_rs_pred_bench'] = _new_pred_bench.copy()
                    st.rerun()
            with _pred_run_btn_c2:
                if _pred_done_sig is None:
                    st.caption('Press **▶ Run predictions** to apply the trained models to this dataset.')
                elif _pred_stale:
                    st.warning(
                        '⚠ Config changed since the last prediction run. Press '
                        '**▶ Run predictions** to refresh — plots below show the previous run.'
                    )
                else:
                    _prev_pb = st.session_state.get('_pred_done_bench', pd.DataFrame())
                    st.caption(f'✓ Showing the last prediction run ({len(_prev_pb):,} row(s)).')

            # Read result from session state
            pred_bench = st.session_state.get('_pred_done_bench', pd.DataFrame())
            if pred_bench is None:
                pred_bench = pd.DataFrame()
            if pred_bench.empty:
                st.warning('No complete prediction rows could be generated. Check that the uploaded data contains the required geochemistry columns for the trained models.')
            else:
                st.session_state['_rs_pred_bench']=pred_bench.copy()
                st.subheader('Prediction summary')
                pred_stats=prediction_summary_stats(pred_bench)
                if not pred_stats.empty:
                    st.caption('Distribution of model predictions across uploaded samples. See the Result Summary tab for group-split violin plots.')
                    table_action_card('Prediction summary by model',pred_stats,'prediction_summary.csv','uk_prediction_summary')
                uk_show_ci=st.checkbox('Tree 90% CI bars',False,key='uk_show_ci',help='Show ensemble uncertainty: 90% interval across individual decision trees (ExtraTrees / RandomForest only)')
                if uk_show_ci and 'Predicted_CI90_Low_km' not in pred_bench:
                    st.info('Tree CI requires ExtraTrees or RandomForest — not available for boosting algorithms.')

                # "Group rows by" picker — same pattern as Validate. Lets
                # the user pick any categorical column on the bench (Arc,
                # Segment, Group_Name, Tectonic_Setting, …) as the
                # grouping axis without needing to re-apply in the Group
                # tab. Drives the Show-groups filter, the per-group
                # averages table, and the per-group overlay on the
                # Validation plot.
                def _pred_group_col_candidates(df):
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        return []
                    cands = []
                    _excl = {'Sample_ID', 'Sample_Name', 'Sample', 'Lat', 'Lon',
                             'Latitude', 'Longitude', 'Predicted_km'}
                    for c in df.columns:
                        if c in _excl or str(c).startswith('_'):
                            continue
                        s = df[c].dropna()
                        if s.empty:
                            continue
                        n_unique = s.nunique()
                        if 2 <= n_unique <= 300:
                            if pd.api.types.is_numeric_dtype(s):
                                if not (c.endswith('_ID') or c == 'Group_ID'
                                        or n_unique <= 30):
                                    continue
                            cands.append(c)
                    # See _val_group_col_candidates for rationale: when
                    # both Arc and Segment exist, the synthesised
                    # Arc_or_Segment union is redundant in the picker.
                    if 'Arc' in cands and 'Segment' in cands and 'Arc_or_Segment' in cands:
                        cands = [c for c in cands if c != 'Arc_or_Segment']
                    _priority = ['Group_Name', 'Group_ID', 'Arc', 'Segment',
                                 'Tectonic_Setting', 'Geologic_Domain',
                                 'Geologic_Era', 'Geologic_Period',
                                 'Geologic_Epoch', 'Geologic_Age_Label',
                                 'Rock_Type_Model', 'Dataset', 'Country',
                                 'Province', 'Terrane', 'Formation', 'Model',
                                 'Arc_or_Segment']
                    out = [c for c in _priority if c in cands]
                    out += [c for c in cands if c not in out]
                    return out

                _pred_group_candidates = _pred_group_col_candidates(pred_bench)
                _pred_group_col = None
                _pred_visible_groups = None
                _pred_all_groups = []
                _pred_bench_grouped = pred_bench  # bench with composite col attached
                if _pred_group_candidates:
                    _pred_group_default = next(
                        (c for c in ('Group_Name', 'Group_ID') if c in _pred_group_candidates),
                        _pred_group_candidates[0],
                    )
                    _pgp1, _pgp2, _pgp3 = st.columns([1.4, 1.4, 0.8])
                    _pred_group_col = _pgp1.selectbox(
                        'Group rows by',
                        _pred_group_candidates,
                        index=_pred_group_candidates.index(
                            st.session_state.get('_pred_group_col', _pred_group_default)
                            if st.session_state.get('_pred_group_col', _pred_group_default) in _pred_group_candidates
                            else _pred_group_default
                        ),
                        key='_pred_group_col',
                        format_func=group_picker_label,
                        help='Primary grouping axis on the prediction bench — '
                             'Group_Name (set in the Group tab) or any raw '
                             'categorical column (Arc, Segment, Geologic_Period, …). '
                             'Geologic_* columns share a coarse→fine hierarchy '
                             '(Era ⊃ Period ⊃ Epoch ⊃ Age_Label).',
                    )
                    # Secondary composite picker — same UX as Validate
                    _pred_secondary_choices = ['(none)']
                    if 'Age_Ma' in pred_bench.columns and pred_bench['Age_Ma'].notna().any():
                        _pred_secondary_choices.append('Age bins (Ma)')
                    _pred_secondary_choices += [c for c in _pred_group_candidates if c != _pred_group_col]
                    _pred_secondary_choice = _pgp2.selectbox(
                        'And also by (composite)',
                        _pred_secondary_choices,
                        index=0, key='_pred_group_col_secondary',
                        format_func=group_picker_label,
                        help='Optional second axis. Set to a categorical column '
                             '(Geologic_Period, Tectonic_Setting…) or to "Age bins '
                             '(Ma)" to slice numeric Age_Ma at the chosen width — '
                             'lets you compare predictions through time × space.',
                    )
                    _pred_age_bin_ma = None
                    if _pred_secondary_choice == 'Age bins (Ma)':
                        _pred_age_bin_ma = _pgp3.slider(
                            'Bin width (Ma)', 1, 200, 10, 1,
                            key='_pred_age_bin_ma',
                        )
                    _pred_secondary_col_arg = (None if _pred_secondary_choice in ('(none)', 'Age bins (Ma)')
                                                 else _pred_secondary_choice)
                    _pred_bench_grouped, _pred_effective_group_col = composite_group_column(
                        pred_bench, _pred_group_col,
                        secondary_col=_pred_secondary_col_arg,
                        age_bin_ma=(int(_pred_age_bin_ma) if _pred_age_bin_ma else None),
                        dest_col='_Composite_Group',
                    )
                    if _pred_effective_group_col != _pred_group_col:
                        _pgp3.caption(f'**{int(_pred_bench_grouped[_pred_effective_group_col].nunique())}** composite groups')
                    _pred_group_col = _pred_effective_group_col
                    _pred_all_groups = sorted(
                        str(g) for g in _pred_bench_grouped[_pred_group_col].dropna().astype(str).unique()
                    )
                    _pgv1, _pgv2 = st.columns([4, 1])
                    _pred_visible_groups = _pgv1.multiselect(
                        f'Show groups (`{_pred_group_col}`)',
                        _pred_all_groups,
                        default=st.session_state.get(
                            f'_pred_visible_groups::{_pred_group_col}',
                            _pred_all_groups,
                        ),
                        key=f'_pred_visible_groups::{_pred_group_col}',
                        help='Untick a group to hide its points and any best-fit '
                             'line for that group. Cached predictions are not '
                             're-run — toggling is instant.',
                    )
                    if _pgv2.button('Reset', key=f'_pred_visible_groups_reset::{_pred_group_col}',
                                    help='Re-show every group'):
                        st.session_state[f'_pred_visible_groups::{_pred_group_col}'] = _pred_all_groups
                        st.rerun()

                # Apply the per-group filter before plotting; mirrors the
                # _plot_bench pattern on Validate so the Validation plot
                # respects the user's group selection. Use the composite-
                # augmented bench when a grouping is in play, else the raw
                # pred_bench.
                _plot_pred_bench = (_pred_bench_grouped
                                    if (_pred_group_col is not None
                                        and isinstance(_pred_bench_grouped, pd.DataFrame)
                                        and not _pred_bench_grouped.empty)
                                    else pred_bench)
                if (_pred_visible_groups is not None
                        and _pred_group_col in _plot_pred_bench.columns
                        and len(_pred_visible_groups) < len(_pred_all_groups)):
                    _gn_pred_str = _plot_pred_bench[_pred_group_col].astype(str)
                    _plot_pred_bench = _plot_pred_bench[_gn_pred_str.isin(_pred_visible_groups)].copy()
                    st.caption(
                        f'Showing **{len(_pred_visible_groups)} of {len(_pred_all_groups)}** '
                        f'`{_pred_group_col}` group(s) '
                        f'({len(_plot_pred_bench):,} of {len(pred_bench):,} rows). '
                        'Untick groups in **Show groups** above to filter further.'
                    )

                # ── Validation plot (Predict tab) ─────────────────────────
                # Same picker UI as Validate, minus the Observed_km axis
                # option (no known thickness on prediction data).
                with st.expander('Validation plot', expanded=True):
                    _upm_axis_opts = simple_xy_axis_options(_plot_pred_bench)
                    if not _upm_axis_opts:
                        st.info('No usable columns on this prediction bench yet.')
                    else:
                        # Mirror the Validate-tab caption explaining what
                        # Predicted_km currently resolves to.
                        _pmt_now_uk = st.session_state.get('primary_model_type', 'Machine Learning')
                        if _pmt_now_uk in ('Multi-ratio', 'Single-ratio'):
                            _pm_proxy_uk = st.session_state.get('_primary_proxy_name', 'Proxy estimate')
                            st.caption(f'ℹ️ `Predicted_km` currently = **{_pm_proxy_uk}** (proxy estimate, not an ML prediction). Pick a `Predicted_<model>_km` column to plot a specific trained ML model.')
                        else:
                            _pm_primary_uk = st.session_state.get('active_interp_model', '')
                            if _pm_primary_uk:
                                st.caption(f'ℹ️ `Predicted_km` = **{_pm_primary_uk}** (primary model). Per-model columns `Predicted_<model>_km` let you pick a specific trained model on either axis.')
                        # Primary-model per-model column for default Y.
                        _pm_safe_uk = _sanitise_model_name(st.session_state.get('active_interp_model', '')) if st.session_state.get('active_interp_model') else ''
                        _pm_y_col_uk = f'Predicted_{_pm_safe_uk}_km' if _pm_safe_uk else ''
                        # Defaults for Predict (no Observed_km available):
                        # X = a proxy ratio if available so the formula
                        # curve has something to draw, else Predicted_km;
                        # Y = primary-model column if present, else Predicted_km;
                        # Colour = Predicted_km.
                        _upm_x_default = next((c for c in ('Sr_Y', 'La_Yb_N', 'Predicted_km')
                                               if c in _upm_axis_opts), _upm_axis_opts[0])
                        _upm_y_default = (
                            _pm_y_col_uk if _pm_y_col_uk in _upm_axis_opts else
                            ('Predicted_km' if 'Predicted_km' in _upm_axis_opts else _upm_axis_opts[0])
                        )
                        _upm_c_default = (
                            _pm_y_col_uk if _pm_y_col_uk in _upm_axis_opts else
                            ('Predicted_km' if 'Predicted_km' in _upm_axis_opts else None)
                        )

                        # Two-stage pickers (group → column) for each axis,
                        # mirroring the Validate-tab layout.
                        _upx_x, _upx_y, _upx_c, _upx_sz = st.columns([1.6, 1.6, 1.6, 0.9])
                        _upm_x = simple_xy_two_stage_picker(
                            'X axis', _upm_axis_opts, _upm_x_default,
                            key_prefix='uk_xy_x', container=_upx_x,
                        )
                        _upm_y = simple_xy_two_stage_picker(
                            'Y axis', _upm_axis_opts, _upm_y_default,
                            key_prefix='uk_xy_y', container=_upx_y,
                        )
                        _upm_c = simple_xy_two_stage_picker(
                            'Colour by', _upm_axis_opts, _upm_c_default,
                            key_prefix='uk_xy_c', container=_upx_c,
                            include_none=True,
                        )
                        _upx_sz.markdown('**Point size**')
                        _upm_pt_size = _upx_sz.slider('Point size', 2, 12, 6, 1,
                                                      key='uk_xy_size',
                                                      label_visibility='collapsed')

                        # Overlays row — show points + reference overlays
                        _ucp0, _ucp1, _ucp2, _ucp3 = st.columns(4)
                        _upm_show_pts   = _ucp0.checkbox('Show points', True, key='uk_xy_show_pts',
                                                         help='Hide the sample scatter and keep only the overlays.')
                        _upm_one_to_one = _ucp1.checkbox('1:1 line', False, key='uk_xy_121',
                                                         help='Only meaningful when both axes are thickness columns')
                        _upm_curve      = _ucp2.checkbox('Formula curve', True, key='uk_xy_curve',
                                                         help='Overlay the proxy calibration when (X, Y) matches a known formula pair')
                        _upm_bestfit    = _ucp3.checkbox('Best fit', False, key='uk_xy_bf')

                        # Per-group statistic overlay — same as Validate.
                        _upm_has_groups = (_pred_group_col is not None
                                            and _pred_group_col in _plot_pred_bench.columns
                                            and _plot_pred_bench[_pred_group_col].notna().any())
                        _upm_group_stat = 'Off'
                        if _upm_has_groups:
                            _upg1, _upg2 = st.columns([1.0, 3.0])
                            _upm_group_stat = _upg1.selectbox(
                                'Per-group overlay',
                                ['Off', 'Median', 'Mean', 'Both'],
                                index=0, key='uk_xy_group_stat',
                                help='Plot one diamond (median) or star (mean) per group at '
                                     'the centre of that group\'s scatter.',
                            )
                            _upg2.caption(
                                'Diamonds = group medians · stars = group means. '
                                'One marker per `Group_Name`.'
                            )

                        # Moving-summary + clip row
                        _ukt1, _ukt2, _ukt3, _ukt4 = st.columns([1.2, 1.0, 1.0, 1.0])
                        _upm_trend     = _ukt1.selectbox(
                            'Moving summary', ['Off', 'Median', 'Mean', 'Both'],
                            index=0, key='uk_xy_trend',
                        )
                        _upm_trend_bin = _ukt2.slider('Bin width', 0.5, 20.0, 5.0, 0.5,
                                                      key='uk_xy_trend_bin')
                        _upm_trend_n   = _ukt3.slider('Min N',     1, 50,   5,  1,
                                                      key='uk_xy_trend_min_n')
                        _uk_clip_opts  = {'None': 0.0, '0.5%': 0.5, '1%': 1.0, '2%': 2.0, '5%': 5.0}
                        _upm_clip      = _uk_clip_opts[_ukt4.selectbox(
                            'Clip outliers', list(_uk_clip_opts), index=0, key='uk_xy_clip',
                        )]

                        if 'H_GAME_LuffiDucea2022_km' in _plot_pred_bench.columns:
                            render_game_settings_card(key_suffix='_uk')

                        if _upm_x is None or _upm_y is None:
                            st.info('Pick both an X and a Y column to render the chart.')
                            _upm_fig = None
                        else:
                            _upm_fig = simple_xy_figure(
                                _plot_pred_bench, _upm_x, _upm_y, _upm_c,
                                show_one_to_one=_upm_one_to_one,
                                show_curve=_upm_curve,
                                show_best_fit=_upm_bestfit,
                                point_size=int(_upm_pt_size),
                                height=540,
                                trend_stat=_upm_trend,
                                trend_bin_width=float(_upm_trend_bin),
                                trend_min_n=int(_upm_trend_n),
                                clip_pct=float(_upm_clip),
                                show_points=bool(_upm_show_pts),
                                group_stat=str(_upm_group_stat),
                                group_col=(_pred_group_col or 'Group_Name'),
                            )
                        if _upm_fig is None:
                            st.warning('Pick columns that have at least one row of overlapping numeric data.')
                        else:
                            st.plotly_chart(_upm_fig, width='stretch')
                            _uk_formula_text = PROXY_THICKNESS_FORMULAS.get(_upm_y) \
                                               or PROXY_THICKNESS_FORMULAS.get(_upm_x)
                            if _upm_curve and _uk_formula_text:
                                st.caption(f'**Formula in view:** {_uk_formula_text}')

                with st.expander('Sample size adequacy', expanded=False):
                    st.caption(
                        'How many samples from your target area do you need for the median prediction to be stable? '
                        'The chart bootstraps the median at each n using your uploaded samples and shows the '
                        'confidence interval narrowing as n grows. '
                        'Green and orange bands show ±5 km and ±10 km precision targets — adjust below.'
                    )
                    _uk_ss_c1, _uk_ss_c2, _uk_ss_c3, _uk_ss_c4 = st.columns(4)
                    _uk_ss_model_opts = pred_bench['Model'].dropna().astype(str).unique().tolist() if 'Model' in pred_bench else list(models.keys())
                    _uk_ss_model = _uk_ss_c1.selectbox('Model', _uk_ss_model_opts or ['(none)'], index=0, key='uk_ss_model')
                    _uk_ss_ci = _uk_ss_c2.selectbox('Confidence interval (%)', [80, 90, 95, 99], index=2, key='uk_ss_ci')
                    _uk_ss_t1 = _uk_ss_c3.number_input('Precision target 1 (±km)', 1.0, 30.0, 5.0, 0.5, key='uk_ss_t1')
                    _uk_ss_t2 = _uk_ss_c4.number_input('Precision target 2 (±km)', 1.0, 30.0, 10.0, 0.5, key='uk_ss_t2')
                    _uk_ss_vals = pd.to_numeric(
                        pred_bench.loc[pred_bench['Model'].astype(str) == str(_uk_ss_model), 'Predicted_km'],
                        errors='coerce'
                    ).dropna().values
                    render_sample_size_panel(_uk_ss_vals, ci_pct=int(_uk_ss_ci),
                                             precision_targets=(_uk_ss_t1, _uk_ss_t2),
                                             key_prefix='uk_ss')

                # GAME diagnostics — trimmed on Predict in parallel with
                # Validate. Consensus tab dropped (the Validation plot does
                # the GAME-vs-Predicted scatter); kept Reliability +
                # Calibration which carry the unique N-survival tracking
                # and per-sensor LOWESS surface.
                if 'H_GAME_LuffiDucea2022_km' in pred_bench:
                    with st.expander('GAME diagnostics',expanded=False):
                        st.caption(
                            'For the GAME-vs-prediction scatter use the **Validation plot** above '
                            'with `H_GAME_LuffiDucea2022_km` on the Y axis. This expander '
                            'keeps the diagnostics that view can\'t reproduce: per-sensor '
                            'calibration surfaces and N-mohometer survival tracking.'
                        )
                        uk_gtab2,uk_gtab3=st.tabs(['Reliability','Calibration'])
                        with uk_gtab2:
                            st.caption('GAME explicitly tracks how many mohometers survive data-availability, reference-model residual/RMSE, and STD/MAD filtering. Low kept N, high MAD/IQR, wide bootstrap CI, or an "all valid; high spread" status should be treated as lower-confidence interpretation.')
                            uk_game_rel_size=int(st.session_state.get('global_point_size', 5))
                            uk_rfig=game_reliability_figure(pred_bench,'Predicted_km',uk_game_rel_size)
                            if uk_rfig is not None:
                                st.plotly_chart(uk_rfig,width='stretch',key='uk_game_reliability_fig')
                            uk_rel_counts=pred_bench['GAME_Luffi2022_Reliability'].fillna('unknown').value_counts().rename_axis('Reliability').reset_index(name='Rows') if 'GAME_Luffi2022_Reliability' in pred_bench else pd.DataFrame()
                            if not uk_rel_counts.empty:
                                table_action_card('GAME reliability counts',uk_rel_counts,'uk_game_reliability_counts.csv','uk_game_reliability_counts')
                            uk_rel_summary=game_reliability_summary(pred_bench,'Predicted_km')
                            if not uk_rel_summary.empty:
                                table_action_card('GAME reliability by N',uk_rel_summary,'uk_game_reliability_by_n.csv','uk_game_reliability_by_n')
                            uk_status_counts=pred_bench['GAME_Luffi2022_Status'].fillna('unknown').value_counts().rename_axis('Status').reset_index(name='Rows') if 'GAME_Luffi2022_Status' in pred_bench else pd.DataFrame()
                            if not uk_status_counts.empty:
                                table_action_card('GAME status counts',uk_status_counts,'uk_game_status_counts.csv','uk_game_status_counts')
                        with uk_gtab3:
                            uk_sensor_options=[s for s,_,_ in GAME_SENSORS if s in game_calibrators()]
                            if uk_sensor_options:
                                uk_sensor=st.selectbox('Mohometer calibration',uk_sensor_options,index=0,key='uk_game_calibration_sensor',format_func=lambda s: game_sensor_labels().get(s,s))
                                uk_cfig=game_calibration_figure(pred_bench,uk_sensor)
                                if uk_cfig is not None:
                                    st.plotly_chart(uk_cfig,width='stretch',key='uk_game_calibration_fig')
                            table_action_card('GAME mohometer summary',game_sensor_summary(),'uk_game_mohometer_summary.csv','uk_game_mohometer_summary')

                # Legacy "Proxy comparison" block removed — replaced by
                # the unified Validation plot above. Same proxy/multi-
                # method scatter, formula curves, moving summary and
                # outlier clipping in a single picker-driven view.

                # ── Per-group model assignment ───────────────────────────────
                # Hidden in 'single' mode — there's only one model, so
                # routing per group makes no sense.
                if (_pred_model_mode == 'multi'
                        and models and not uk_raw.empty
                        and 'Group_ID' in uk_raw.columns):
                    with st.expander('🔬 Per-group model assignment', expanded=True):
                        st.caption('This dataset has group labels — assign a different trained model to each group and run per-group predictions.')
                        per_group_model_panel(uk_raw, 'predict', _pred_active_models, la_mode, '_rs_pred_bench')

                st.caption('Download predictions in the Summary tab → Export results.')

# ─────────────────────────────────────────────────────────────────────────────
# Group tab — define groups via stacked table filters with classification
# options (equal / geometric / quantile / Jenks). Lasso selection on map +
# graph and user-drawn polyline-as-axis come in subsequent landings.
# ─────────────────────────────────────────────────────────────────────────────
with t_grouping:
    st.header('Group')
    st.caption('Define groups by stacking categorical and numeric filters from the table, '
               'lasso-selecting on the map or cross-plot, or enabling the PCA polyline '
               'projection for along-strike analysis.')

    # --- Source selector ----------------------------------------------------
    # Reference datasets (built-ins + custom refs) are first-class options
    # here — no need to load them through Prepare first. Pool entries from
    # Prepare uploads are listed alongside, filtered by role assignment.
    _pool_for_group = get_prepped_pool()
    _dp_roles       = st.session_state.get('_dp_dataset_roles', {})
    _dp_sheet_keys  = list(st.session_state.get('_dp_sheet_pool_keys', []))

    _ROLE_TO_TAB = {
        'Training':   'Model (training)',
        'Validation': 'Validate',
        'Prediction': 'Predict',
    }
    _g_role_label = st.radio(
        'Dataset role', list(_ROLE_TO_TAB.keys()),
        horizontal=True, key='grouping_role',
        help='Pick which role this grouping is for. References (Guo / Zou / '
             'Luffi / your custom refs) and Prepare-tab uploads assigned this '
             'role both appear below. Group_ID gets written back to the '
             'chosen dataset, so the matching downstream tab (Model / '
             'Validate / Predict) sees the groups.',
    )
    _g_target_role = _ROLE_TO_TAB[_g_role_label]

    # Build the option list:
    #   1. Reference datasets — always visible (role-agnostic; references
    #      are useful for any role and don't carry role assignments).
    #   2. Pool entries from Prepare — gated by role. Stems matching a
    #      built-in's filename are hidden (the built-in friendly name is
    #      already in the list and read_training_source prefers the
    #      prepared pool entry when one exists).
    _ref_options = [r for r in available_reference_labels()]
    _builtin_stems_in_use = {
        _BUILTIN_TO_FILE_STEM[r] for r in _ref_options
        if r in _BUILTIN_TO_FILE_STEM
    }
    _pool_options = []
    for _pn in _dp_sheet_keys:
        if _pn in _builtin_stems_in_use:
            continue   # already covered by friendly built-in name
        if _pn not in _pool_for_group:
            continue
        # Only include pool entries assigned the chosen role.
        if _g_target_role not in _dp_roles.get(_pn, set()):
            continue
        # Skip already-grouped (re-grouping is circular)
        if 'Group_ID' in _pool_for_group[_pn].columns:
            continue
        _pool_options.append(_pn)
    # Group-derived entries (e.g. 'GuoYang_2023_Model [grouped]') are also
    # excluded — they're already grouped by definition.
    _g_dataset_options = _ref_options + _pool_options
    if not _g_dataset_options:
        st.info(
            'No datasets available. Either click a reference button above '
            f'(Guo / Zou / Luffi) in the **Prepare** tab\'s reference '
            f'loader, or upload a file in Prepare and assign it the '
            f'**{_g_role_label}** role.'
        )
        st.stop()

    _g_src = st.selectbox(
        f'{_g_role_label} dataset', _g_dataset_options,
        key=f'grouping_source_{_g_target_role}',
        help='Pick the dataset to define groups on. References load directly; '
             'pool entries from Prepare appear when assigned this role.',
    )
    # Resolve via the unified loader — handles all three cases (reference
    # by friendly name, pool entry by stem/custom name, group-derived).
    _g_bench = load_dataset_by_name(_g_src)
    if _g_bench.empty:
        st.warning(f"'{_g_src}' could not be loaded. If it's a custom "
                   "reference, the file path may have moved.")
        st.stop()
    _g_bench = _g_bench.reset_index(drop=True)
    _g_value_default_candidates = ['Predicted_km', 'Observed_km', 'Crust_Thickness']
    _g_num_cols_all = list(_g_bench.select_dtypes('number').columns)
    _g_value_default = next(
        (c for c in _g_value_default_candidates if c in _g_bench.columns),
        _g_num_cols_all[0] if _g_num_cols_all else None
    )

    # --- Filter editor ------------------------------------------------------
    st.markdown('**Filters** — each row narrows or partitions the dataset. AND-combined.')
    _filter_state_key = f'_g_filters_{_g_src}'
    if _filter_state_key not in st.session_state:
        st.session_state[_filter_state_key] = []  # list of dicts

    # Cache column discovery — only recompute when the bench actually changes
    # (not on every slider/widget interaction, which makes the tab very slow).
    _bench_sig = _g_src + ':' + ':'.join(str(c) for c in _g_bench.columns) + ':' + str(len(_g_bench))
    if st.session_state.get('_g_bench_sig') != _bench_sig:
        st.session_state['_g_bench_sig'] = _bench_sig
        st.session_state['_g_cat_cols_c'] = _g_candidate_group_columns(_g_bench)
        st.session_state['_g_num_cols_c'] = _g_candidate_numeric_columns(_g_bench)
    _cat_cols = list(st.session_state.get('_g_cat_cols_c', []))
    _num_cols = list(st.session_state.get('_g_num_cols_c', []))
    # Apply Prepare-tab column-type overrides if available
    _prep_types: dict = st.session_state.get('dp_column_types', {})
    if _prep_types:
        _bench_cols_set = set(_g_bench.columns)
        # Columns the user typed as 'Metadata' → remove from both lists
        _meta_cols = {c for c, t in _prep_types.items() if t == 'Metadata' and c in _bench_cols_set}
        _cat_cols = [c for c in _cat_cols if c not in _meta_cols]
        _num_cols = [c for c in _num_cols if c not in _meta_cols]
        # Columns typed as 'Category' → ensure they appear in _cat_cols, not _num_cols
        _user_cat_cols = [c for c, t in _prep_types.items()
                          if t == 'Category' and c in _bench_cols_set and c not in _cat_cols]
        _cat_cols = _user_cat_cols + _cat_cols
        _num_cols = [c for c in _num_cols if _prep_types.get(c) != 'Category']
        # Columns typed as 'Numeric' (unmapped keep-original) → ensure in _num_cols if not already
        _user_num_cols = [c for c, t in _prep_types.items()
                          if t == 'Numeric' and c in _bench_cols_set and c not in _num_cols
                          and c not in _meta_cols and c not in _cat_cols]
        _num_cols = _num_cols + _user_num_cols

    _af1, _af2, _af3 = st.columns([1.2, 1.2, 0.7])
    _add_kind = _af1.selectbox('Add filter type',
                               ['Categorical (multi-select)',
                                'Numeric range',
                                'Numeric bins (partition)'],
                               key='g_add_kind')
    if _add_kind.startswith('Categorical'):
        _add_col = _af2.selectbox('Column', _cat_cols, key='g_add_cat_col') if _cat_cols else None
    elif _add_kind.startswith('Numeric range'):
        _add_col = _af2.selectbox('Column', _num_cols, key='g_add_range_col') if _num_cols else None
    else:
        _add_col = _af2.selectbox('Column', _num_cols, key='g_add_bin_col') if _num_cols else None

    if _af3.button('+ Add', key='g_add_filter') and _add_col is not None:
        if _add_kind.startswith('Categorical'):
            spec = {'type': 'categorical', 'col': _add_col, 'values': None}  # None → show all on first render
        elif _add_kind.startswith('Numeric range'):
            ser = pd.to_numeric(_g_bench[_add_col], errors='coerce')
            spec = {'type': 'numeric_range', 'col': _add_col,
                    'lo': float(ser.min()) if ser.notna().any() else 0.0,
                    'hi': float(ser.max()) if ser.notna().any() else 1.0}
        else:
            spec = {'type': 'numeric_bin', 'col': _add_col,
                    'method': 'equal', 'n_bins': 4, 'keep': []}
        st.session_state[_filter_state_key].append(spec)

    # Render existing filters
    _filters = st.session_state[_filter_state_key]
    _delete_idx = None
    for _fi, _spec in enumerate(list(_filters)):
        with st.container(border=True):
            _r1, _r2, _r3, _r4 = st.columns([1.5, 2.3, 1.0, 0.4])
            _col = _spec.get('col', '')
            _r1.markdown(f"**{_col}** &nbsp; *({_spec.get('type','')})*")
            if _spec['type'] == 'categorical':
                _opts = sorted(_g_bench[_col].dropna().astype(str).unique().tolist()) if _col in _g_bench else []
                # None sentinel = newly added filter → default to all values so groups appear immediately
                _cat_default = _opts if _spec.get('values') is None else _spec.get('values', _opts)
                _spec['values'] = _r2.multiselect('Keep values', _opts, default=_cat_default,
                                                   key=f'g_filter_cat_{_fi}', label_visibility='collapsed')
                _r3.caption(f"{len(_spec['values'])} of {len(_opts)} selected")
            elif _spec['type'] == 'numeric_range':
                _ser = pd.to_numeric(_g_bench[_col], errors='coerce')
                _lo_d = float(_ser.min()) if _ser.notna().any() else 0.0
                _hi_d = float(_ser.max()) if _ser.notna().any() else 1.0
                if _lo_d == _hi_d:
                    _hi_d = _lo_d + 1.0
                _step = max((_hi_d - _lo_d) / 100.0, 1e-3)
                _rng = _r2.slider(f'{_col} range', _lo_d, _hi_d,
                                  (float(_spec.get('lo', _lo_d)), float(_spec.get('hi', _hi_d))),
                                  step=_step, key=f'g_filter_range_{_fi}', label_visibility='collapsed')
                _spec['lo'], _spec['hi'] = float(_rng[0]), float(_rng[1])
                _r3.caption(f"{_spec['lo']:.2f} – {_spec['hi']:.2f} · narrows rows, add a categorical or bins filter to split into named groups")
            elif _spec['type'] == 'numeric_bin':
                _bm1, _bm2 = _r2.columns(2)
                _spec['method'] = _bm1.selectbox('Method', ['equal', 'geometric', 'quantile', 'jenks'],
                                                 index=['equal','geometric','quantile','jenks'].index(_spec.get('method','equal')),
                                                 key=f'g_filter_bin_method_{_fi}',
                                                 format_func=lambda m: {'equal':'Equal interval','geometric':'Geometric',
                                                                         'quantile':'Quantile','jenks':'Natural breaks (Jenks)'}[m])
                _spec['n_bins'] = int(_bm2.number_input('Bins', 2, 12, int(_spec.get('n_bins', 4)),
                                                        step=1, key=f'g_filter_bin_n_{_fi}'))
                _ser = pd.to_numeric(_g_bench[_col], errors='coerce')
                _breaks = _g_numeric_breaks(_ser.dropna().to_numpy(), _spec['method'], _spec['n_bins'])
                _labels_full = _g_assign_numeric_bins(_ser, _breaks)
                _bin_options = sorted(_labels_full.dropna().unique().tolist())
                if 'unknown' in _bin_options:
                    _bin_options.remove('unknown')
                _spec['keep'] = _r3.multiselect('Keep bins', _bin_options,
                                                default=_spec.get('keep', _bin_options),
                                                key=f'g_filter_bin_keep_{_fi}',
                                                label_visibility='collapsed')
            if _r4.button('✕', key=f'g_filter_del_{_fi}', help='Remove this filter'):
                _delete_idx = _fi

    if _delete_idx is not None:
        st.session_state[_filter_state_key].pop(_delete_idx)
        st.rerun()

    if st.button('Clear all filters and groups', key='g_clear_filters'):
        st.session_state[_filter_state_key] = []
        st.session_state.pop(f'_g_groups_{_g_src}', None)
        st.rerun()

    # --- Resolve groups from filters ---------------------------------------
    # Computation is gated behind ▶ Apply groups so that tweaking sliders /
    # multiselects doesn't re-scan the bench on every keystroke.
    # A content-hash (MD5 of bench signature + filter JSON) tells us when
    # the displayed result is stale so we can warn the user.
    import hashlib as _hl_g
    _filt_spec  = st.session_state[_filter_state_key]
    _bench_id   = (
        f'{_g_src}|{len(_g_bench)}|'
        f'{",".join(map(str, _g_bench.columns[:50]))}'
    )
    _filt_sig = _hl_g.md5(
        (_bench_id + '||' + json.dumps(_filt_spec, default=str, sort_keys=True))
        .encode('utf-8')
    ).hexdigest()
    _grouped_cache_key = f'_g_grouped_cache_{_g_src}'
    _grouped_sig_key   = f'{_grouped_cache_key}_sig'
    _sig_stale = (st.session_state.get(_grouped_sig_key) != _filt_sig)
    _apply_col, _stale_col = st.columns([0.22, 0.78])
    _apply_groups = _apply_col.button('▶ Apply groups', key='g_apply_groups', type='primary')
    if _sig_stale and not _apply_groups:
        _stale_col.caption('⚠️ Filters changed — click **▶ Apply groups** to update.')
    if _apply_groups or _grouped_cache_key not in st.session_state:
        st.session_state[_grouped_cache_key] = _g_auto_groups_from_filters(
            _g_bench, _filt_spec,
        )
        st.session_state[_grouped_sig_key] = _filt_sig
    _grouped = st.session_state[_grouped_cache_key].copy()
    _n_passing = int(_grouped['Group_ID'].notna().sum())
    _n_total = len(_grouped)
    _n_groups = int(_grouped['Group_ID'].dropna().nunique())

    _summary = (_g_group_stats(_grouped, value_col=_g_value_default, decimals=2)
                if _g_value_default else pd.DataFrame())

    # --- Group preview table rendered AFTER lasso groups are merged in ------
    # (table appears below, once lasso state has been applied to _grouped)

    # --- Lasso state (Landing 2) -------------------------------------------
    # _g_lasso_groups_{src}: dict[lasso_id, {'name', 'color', 'members'}]
    # 'members' is a set of original-bench row indices.
    _lasso_state_key = f'_g_lasso_groups_{_g_src}'
    if _lasso_state_key not in st.session_state:
        st.session_state[_lasso_state_key] = {}

    # Apply lasso group memberships ON TOP of filter-resolved groups.
    # Lasso assignments override filter assignments (last-write-wins).
    _lasso_palette = px.colors.qualitative.Bold
    for _li, (_lid, _ldef) in enumerate(st.session_state[_lasso_state_key].items()):
        _members = _ldef.get('members', set())
        if not _members:
            continue
        _hits = _grouped.index.isin(list(_members))
        if _hits.any():
            _grouped.loc[_hits, 'Group_ID'] = _lid
            _grouped.loc[_hits, 'Group_Name'] = _ldef.get('name', _lid)
            _grouped.loc[_hits, 'Group_Source'] = 'lasso'

    # Recompute counts and summary so the table reflects lasso groups too
    _n_passing = int(_grouped['Group_ID'].notna().sum())
    _n_groups = int(_grouped['Group_ID'].dropna().nunique())
    _summary = (_g_group_stats(_grouped, value_col=_g_value_default, decimals=2)
                if _g_value_default else pd.DataFrame())

    # --- Unified groups table (filter-based + lasso merged) -----------------
    _has_lasso = bool(st.session_state[_lasso_state_key])
    _disable_key = f'_g_disabled_{_g_src}'
    _disabled_gids: set = set(st.session_state.get(_disable_key, set()))

    if not _summary.empty:
        _n_shown = int(_grouped['Group_ID'].dropna().nunique())
        _n_incl  = _n_shown - sum(1 for gid in _summary['Group_ID'] if gid in _disabled_gids)
        st.markdown(
            f"**Groups** — {_n_shown} group(s), {_n_incl} included "
            f"· {_n_passing:,} of {_n_total:,} rows pass filters"
        )
        if _has_lasso:
            st.caption(
                f"_{len(st.session_state[_lasso_state_key])} lasso group(s) merged in "
                f"— see panel below._"
            )
        # Build display table: Include checkbox + Adequate? badge
        _summary_disp = _summary.copy()
        _summary_disp.insert(0, 'Include',
                             [gid not in _disabled_gids for gid in _summary_disp['Group_ID']])
        _summary_disp.insert(3, 'Adequate?',
                             _summary_disp['N'].apply(
                                 lambda n: '✅' if n >= 20 else ('⚠️' if n >= 10 else '❌')))
        _edited_summary = st.data_editor(
            _summary_disp,
            column_config={
                'Include': st.column_config.CheckboxColumn(
                    'Include',
                    help='Uncheck to disable this group — excluded from aggregation '
                         'and downstream use',
                    default=True,
                )
            },
            disabled=[c for c in _summary_disp.columns if c != 'Include'],
            hide_index=True,
            use_container_width=True,
            key='g_group_table_editor',
        )
        # Persist Include state → disabled set
        _new_disabled = {
            row['Group_ID']
            for _, row in _edited_summary.iterrows()
            if not row['Include']
        }
        st.session_state[_disable_key] = _new_disabled
        _disabled_gids = _new_disabled
        # Apply disabled groups — wipe Group_ID so they're invisible downstream
        if _disabled_gids:
            _mask_dis = _grouped['Group_ID'].isin(_disabled_gids)
            _grouped.loc[_mask_dis, 'Group_ID']   = np.nan
            _grouped.loc[_mask_dis, 'Group_Name'] = np.nan
            _n_passing = int(_grouped['Group_ID'].notna().sum())
            _n_groups  = int(_grouped['Group_ID'].dropna().nunique())
    else:
        st.markdown('**Groups** — no groups defined yet')
        st.info('Add at least one filter above, or drag a lasso/box on the map or '
                'cross-plot, to define groups.')

    # --- Linked map + cross-plot -------------------------------------------
    # Always render when the bench has Lat/Lon, even before any groups exist —
    # lasso-grouping needs the map visible from the start so the user can drag
    # a selection without first adding a filter category.  Ungrouped rows are
    # drawn in neutral grey; once a filter or lasso group is added, those rows
    # get a colour and the legend follows.
    if {'Lat', 'Lon'}.issubset(_grouped.columns):
        # ── Landing 3: Polyline-as-axis ──────────────────────────────────────
        with st.expander('🔬 Polyline axes — PCA long-axis projection', expanded=False):
            st.checkbox(
                'Enable per-group PCA polyline projection',
                value=st.session_state.get('_g_polyline_on', False),
                key='_g_polyline_on',
                help='Fits a straight-line axis through each group\'s samples via PCA '
                     '(equirectangular km projection). Adds Along_Strike_km and '
                     'Across_Strike_km columns; draws the axis on the map.',
            )
            if st.session_state.get('_g_polyline_on', False):
                _pa1, _pa2 = st.columns(2)
                _pa1.selectbox(
                    'Rolling average statistic',
                    ['median', 'mean'],
                    key='_g_poly_stat',
                    format_func=lambda s: {'median': 'Median + IQR band',
                                           'mean': 'Mean ± 1 SD'}[s],
                )
                _pa2.slider(
                    'Rolling window (km)', 50, 1000, 200, 50,
                    key='_g_poly_window',
                    help='Half-window = window ÷ 2 km on either side of each grid point.',
                )
                st.caption(
                    '**Along_Strike_km** = distance along the group PCA axis from the '
                    'south-west end. **Across_Strike_km** = signed perpendicular distance '
                    '(positive = left of direction vector). Select **Along_Strike_km** as '
                    'the cross-plot X axis to see a rolling-window smoothing curve.'
                )

        # Read polyline settings from session state (survives expander collapsed/open)
        _poly_on = st.session_state.get('_g_polyline_on', False)
        _poly_stat = st.session_state.get('_g_poly_stat', 'median')
        _poly_window = int(st.session_state.get('_g_poly_window', 200))
        _group_axes: dict = {}
        if _poly_on and _n_passing > 0:
            try:
                _grouped_proj, _group_axes = _g_attach_polyline_projections(
                    _grouped, group_col='Group_ID', lon_col='Lon', lat_col='Lat'
                )
                if 'Along_Strike_km' in _grouped_proj.columns:
                    _grouped = _grouped_proj
            except Exception as _poly_err:
                st.warning(f'Polyline projection failed — {_poly_err}')

        st.markdown('**Map + cross-plot** (linked by Group_ID — drag a lasso/box on either to define groups)')
        if _n_passing == 0:
            st.caption(
                'No groups defined yet. Drag a lasso or box on either chart and '
                'click **+ New group** below — or add a filter above to colour '
                'rows by category.'
            )
        # Show *all* bench rows (ungrouped rows in grey) so the user can lasso
        # before any filter has been added.  Once filters or lassos assign
        # Group_IDs, the same rows pick up colour from the palette below.
        _vis_df = _grouped.copy()
        if not _vis_df.empty:
            _vc1, _vc2, _vc3 = st.columns(3)
            _vis_x_opts = [c for c in _num_cols if c in _vis_df.columns]
            # Polyline projection columns go to the front of the picker when available
            for _pcol in ('Across_Strike_km', 'Along_Strike_km'):
                if _pcol in _vis_df.columns and _pcol not in _vis_x_opts:
                    _vis_x_opts = [_pcol] + _vis_x_opts
            _vis_x = _vc1.selectbox('Cross-plot X', _vis_x_opts,
                                     index=_vis_x_opts.index('Sr_Y') if 'Sr_Y' in _vis_x_opts else 0,
                                     key='g_vis_x')
            _vis_y_opts = [c for c in _vis_x_opts if c != _vis_x]
            _vis_y_default = 'Predicted_km' if _vis_y_opts and 'Predicted_km' in _vis_y_opts else (_vis_y_opts[0] if _vis_y_opts else _vis_x)
            _vis_y = _vc2.selectbox('Cross-plot Y', _vis_y_opts,
                                     index=_vis_y_opts.index(_vis_y_default) if _vis_y_default in _vis_y_opts else 0,
                                     key='g_vis_y')
            _vis_overlay = _vc3.multiselect('Overlay group statistic',
                                             ['Mean', 'Median'], default=['Median'],
                                             key='g_vis_overlay')

            # Marker styling — size + border. Kept on a single row so it
            # doesn't push the charts down. Border is OFF by default because
            # the per-marker SVG stroke is what makes Scattergeo crawl past
            # ~10 K points; users with smaller datasets can flick it on for
            # better dot definition.
            _ms1, _ms2, _ms3, _ms4 = st.columns([1.6, 0.9, 1.2, 1.4])
            _g_marker_size = _ms1.slider(
                'Marker size', 3, 14,
                int(st.session_state.get('_g_marker_size', 6)),
                key='_g_marker_size',
                help='Applies to both the map and cross-plot sample markers.',
            )
            _g_marker_border = _ms2.checkbox(
                'Marker border',
                value=bool(st.session_state.get('_g_marker_border', False)),
                key='_g_marker_border',
                help='Draws a thin black outline around each point. Has no '
                     'effect on the WebGL cross-plot (Scattergl), and slows '
                     'the map down on >50 K points.',
            )
            _g_overlay_size = _ms3.slider(
                'Overlay (median/mean) size', 6, 28,
                int(st.session_state.get('_g_overlay_size', 14)),
                key='_g_overlay_size',
                help='Diamond/star markers for the per-group statistic overlay.',
            )
            # Hover scope — the per-sample callout has ~12 fields and
            # spams the screen on dense maps. Three settings:
            #   • All           — both samples and overlay (legacy)
            #   • Overlays only — sample dots silent, only the median/
            #                     mean centroid shows a callout (default)
            #   • Off           — nothing hovers (still selectable via
            #                     lasso/box, just no tooltip)
            _hover_choices = ['All', 'Overlays only', 'Off']
            _g_hover_mode = _ms4.selectbox(
                'Hover',
                _hover_choices,
                index=_hover_choices.index(
                    st.session_state.get('_g_hover_mode', 'Overlays only')
                ) if st.session_state.get('_g_hover_mode', 'Overlays only') in _hover_choices else 1,
                key='_g_hover_mode',
                help='Controls which markers show a hover callout. Sample '
                     'callouts are verbose (~12 fields) and can crowd '
                     'dense charts; "Overlays only" keeps the per-group '
                     'median/mean tooltip and silences individual samples.',
            )

            # Build per-row colour from a stable palette keyed by group name.
            # Ungrouped rows (Group_Name NaN) → neutral grey; only real group
            # names get a palette colour.
            #
            # Lasso groups carry their own committed colour in
            # ``_lg_state[lid]['color']`` — use that first so the chart
            # matches the swatch shown in the "Lasso groups" panel below.
            # Filter-based groups that don't have a stored colour fall back
            # to the same Bold palette the lasso panel uses, keyed by the
            # group's order of appearance, so creation-order is consistent.
            _palette = list(px.colors.qualitative.Bold) + list(px.colors.qualitative.Set2)
            _UNGROUPED_GREY = '#9ca3af'
            _grp_names_in_view = [
                str(g) for g in dict.fromkeys(_vis_df['Group_Name'].dropna().astype(str).tolist())
            ]
            # Map lasso-group-name → stored colour
            _lasso_colour_by_name = {
                str(_ldef.get('name', _lid)): _ldef.get('color')
                for _lid, _ldef in st.session_state.get(_lasso_state_key, {}).items()
                if _ldef.get('color')
            }
            # User-specified colour overrides survive across reruns and
            # apply to ANY group (filter-based or lasso). Keyed by group
            # name, namespaced per source dataset so re-loading a different
            # bench doesn't inherit stale picks.
            _color_override_key = f'_g_color_overrides_{_g_src}'
            if _color_override_key not in st.session_state:
                st.session_state[_color_override_key] = {}
            _user_colour_overrides = st.session_state[_color_override_key]

            _color_map = {}
            _palette_idx = 0
            for gn in _grp_names_in_view:
                if gn in _user_colour_overrides:
                    # Highest priority — explicit user pick
                    _color_map[gn] = _user_colour_overrides[gn]
                elif gn in _lasso_colour_by_name:
                    # Use the colour the lasso panel already assigned
                    _color_map[gn] = _lasso_colour_by_name[gn]
                else:
                    _color_map[gn] = _palette[_palette_idx % len(_palette)]
                    _palette_idx += 1

            # ── Display downsampling ─────────────────────────────────────
            # Plotly's Scattergeo and SVG-based Scatter slow past ~10K points,
            # but the user's expectation is "lasso captures everything inside
            # the polygon" — when we downsample, hidden rows fall through
            # both Plotly's per-point hit-test AND the polygon-coords helper
            # (Scattergeo's lasso events don't reliably ship vertices).
            #
            # Resolution: draw EVERY row by default (Scattergl on the
            # cross-plot handles 100K+; Scattergeo with no marker.line is
            # acceptable to ~50-80K). Power-users can re-enable downsampling
            # via the checkbox if a dataset is so large that interactions
            # slow noticeably — but the default trades a bit of redraw cost
            # for a lasso that actually works.
            _g_downsample_on = st.checkbox(
                'Downsample large datasets for chart speed (lasso may miss '
                'points hidden by sampling)',
                value=False, key='_g_downsample_on',
                help='OFF (default) — every bench row is drawn so the lasso '
                     'captures all interior points. ON — randomly subsample '
                     'ungrouped rows down to 10 000 for faster redraws.',
            )
            _G_DISPLAY_CAP = 10000 if _g_downsample_on else 10**9
            _vis_total_rows = len(_vis_df)
            if _vis_total_rows > _G_DISPLAY_CAP:
                _has_group_mask = _vis_df['Group_Name'].notna()
                _grouped_rows   = _vis_df[_has_group_mask]
                _ungrouped_rows = _vis_df[~_has_group_mask]
                _budget_for_ungrouped = max(0, _G_DISPLAY_CAP - len(_grouped_rows))
                if _budget_for_ungrouped < len(_ungrouped_rows):
                    _ungrouped_sampled = _ungrouped_rows.sample(
                        n=_budget_for_ungrouped, random_state=42,
                    )
                else:
                    _ungrouped_sampled = _ungrouped_rows
                # Order matters for plotly painting: later rows draw ON TOP.
                # Put ungrouped rows FIRST so they paint underneath, and
                # grouped rows LAST so their colours sit on top of any
                # overlapping grey neighbours (otherwise dense clusters
                # like South-American arc samples show as a sea of grey
                # even when 1000+ are actually in a lasso group).
                _vd = pd.concat([_ungrouped_sampled, _grouped_rows])
                _vd_was_downsampled = True
            else:
                # Even without downsampling, sort so grouped rows paint on
                # top of ungrouped neighbours — same z-order rationale.
                _has_group_mask = _vis_df['Group_Name'].notna()
                _vd = pd.concat([
                    _vis_df[~_has_group_mask],
                    _vis_df[_has_group_mask],
                ]) if _has_group_mask.any() else _vis_df.copy()
                _vd_was_downsampled = False

            # ── Cache the heavy display-prep work ───────────────────────
            # Numeric rounding + customdata stack are O(N × cols) and run on
            # every Streamlit rerun (every widget interaction).  Memoize them
            # against a content signature of _vd so they only re-fire when
            # the displayed rows actually change (group additions, filter
            # changes, tab switch with new bench).
            #
            # The signature includes a fingerprint of the lasso state and
            # the per-row Group_Name values — without these, adding/removing
            # a lasso group would silently hit the stale cache and the chart
            # would render all-grey because the cached Group_Name vector
            # doesn't match the freshly-built _color_map keys.
            _lasso_state_for_sig = st.session_state.get(_lasso_state_key, {})
            _lasso_sig_str = json.dumps(
                {str(_lid): {
                    'name':  str(_ldef.get('name', '')),
                    'color': str(_ldef.get('color', '')),
                    'n':     len(_ldef.get('members', set())),
                } for _lid, _ldef in _lasso_state_for_sig.items()},
                sort_keys=True,
            )
            _gnames_fp = ','.join(_grp_names_in_view) or 'no-groups'
            _color_overrides_fp = json.dumps(_user_colour_overrides, sort_keys=True)
            _disp_sig = _hl_g.md5(
                f'{_g_src}|{len(_vd)}|{",".join(map(str, _vd.columns[:80]))}|'
                f'{_vd_was_downsampled}|{_vis_total_rows}|'
                f'{_lasso_sig_str}|{_gnames_fp}|{_color_overrides_fp}'
                .encode('utf-8')
            ).hexdigest()
            _disp_cache_key = f'_g_disp_cache_{_g_src}'
            _disp_cache = st.session_state.get(_disp_cache_key)
            if _disp_cache is None or _disp_cache.get('sig') != _disp_sig:
                # Round numerics to 2dp for display + hover
                _vd_rounded = _vd.copy()
                for c in _vd_rounded.select_dtypes(include='number').columns:
                    _vd_rounded[c] = pd.to_numeric(_vd_rounded[c], errors='coerce').round(2)
                # Hover columns + template
                _hover_cols_list = ['Sample_ID', 'Group_Name', 'Group_Source',
                                    'Predicted_km', 'Observed_km',
                                    'Age_Ma', 'Sr_Y', 'La_Yb_N', 'SiO2', 'MgO',
                                    'H_GAME_LuffiDucea2022_km', 'H_Sundell2021_Paired_km',
                                    'Arc_or_Segment', 'Geologic_Domain', 'Tectonic_Setting']
                _hcp = [c for c in _hover_cols_list if c in _vd_rounded.columns]
                _hover_lines = [f'<b>{_vd_rounded[c].name}=%{{customdata[{i}]}}</b>' if c == 'Sample_ID'
                                else f'{c}=%{{customdata[{i}]}}'
                                for i, c in enumerate(_hcp)]
                _disp_cache = {
                    'sig':        _disp_sig,
                    'vd_rounded': _vd_rounded,
                    'hover_present': _hcp,
                    'hover_tmpl':    '<br>'.join(_hover_lines) + '<extra></extra>',
                    'customdata':    np.column_stack(
                        [_vd_rounded[c].astype(object).to_numpy() for c in _hcp]
                    ) if _hcp else np.empty((len(_vd_rounded), 0)),
                    'orig_index':    _vd_rounded.index.to_numpy(),
                }
                st.session_state[_disp_cache_key] = _disp_cache
            _vd                 = _disp_cache['vd_rounded']
            _hover_cols_present = _disp_cache['hover_present']
            _hover_tmpl         = _disp_cache['hover_tmpl']
            _customdata         = _disp_cache['customdata']
            _vd_orig_index      = _disp_cache['orig_index']
            _row_colors = [
                _UNGROUPED_GREY if pd.isna(g) else _color_map.get(str(g), _UNGROUPED_GREY)
                for g in _vd['Group_Name']
            ]
            if _vd_was_downsampled:
                st.caption(
                    f'Displaying **{len(_vd):,} of {_vis_total_rows:,}** points — '
                    'all grouped points kept; ungrouped points stratified-sampled '
                    'so plot interactions stay snappy. Lasso/box selections still '
                    'land in groups correctly.'
                )

            _mc1, _mc2 = st.columns(2)

            # Map (single trace; lasso point_index → row position in _vd → bench index via _vd_orig_index)
            with _mc1:
                _mfig = go.Figure()
                # marker.line is opt-in (border checkbox above) because
                # the per-marker SVG stroke is what makes Scattergeo crawl
                # past ~10 K points. With border off we get ~5× faster
                # redraws — the right trade-off when the whole point of
                # this view is "draw everything so the lasso works".
                _map_marker_kw = dict(size=int(_g_marker_size), color=_row_colors)
                if _g_marker_border:
                    _map_marker_kw['line'] = dict(color='black', width=0.4)
                # Sample-level hover obeys the Hover selector — silenced
                # for "Overlays only" and "Off" so dense maps stay readable.
                _samples_hover_on = (_g_hover_mode == 'All')
                _mfig.add_trace(go.Scattergeo(
                    lat=_vd['Lat'], lon=_vd['Lon'], mode='markers',
                    marker=_map_marker_kw,
                    customdata=_customdata,
                    hovertemplate=(_hover_tmpl if _samples_hover_on else None),
                    hoverinfo=('skip' if not _samples_hover_on else None),
                    name='samples',
                    showlegend=False,
                ))
                # Landing 3 — axis polylines (one dashed line per group)
                for _ax_gid, _ax_info in _group_axes.items():
                    _ax_gname_rows = _vd.loc[_vd['Group_ID'] == _ax_gid, 'Group_Name'].dropna()
                    _ax_gname = str(_ax_gname_rows.iloc[0]) if not _ax_gname_rows.empty else str(_ax_gid)
                    _ax_color = _color_map.get(_ax_gname, '#333333')
                    _mfig.add_trace(go.Scattergeo(
                        lat=[_ax_info['lat0'], _ax_info['lat1']],
                        lon=[_ax_info['lon0'], _ax_info['lon1']],
                        mode='lines',
                        line=dict(color=_ax_color, width=2.5, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'axis_{_ax_gid}',
                    ))
                # Overlay group centroids on the map (median lat/lon per
                # group). Drawn LAST so the diamonds/stars sit on top of
                # the sample dots and dashed axes — same z-order rationale
                # as the cross-plot overlay.
                if _vis_overlay and not _summary.empty and {'Lat', 'Lon'}.issubset(_vis_df.columns):
                    _centroids = (
                        _vis_df.dropna(subset=['Group_Name', 'Lat', 'Lon'])
                               .groupby('Group_Name', sort=False)
                               .agg(med_lat=('Lat', 'median'),
                                    med_lon=('Lon', 'median'),
                                    n=('Lat', 'size'))
                               .reset_index()
                    )
                    if not _centroids.empty:
                        _cent_colors = [_color_map.get(str(g), '#111827')
                                        for g in _centroids['Group_Name']]
                        # Centroid hover stays on for "All" and "Overlays
                        # only" — the whole point of the median markers
                        # is the per-group summary they expose. Only
                        # "Off" silences them.
                        _ov_hover_on = (_g_hover_mode != 'Off')
                        for _stat in _vis_overlay:
                            _sym = 'diamond' if _stat == 'Median' else 'star'
                            _mfig.add_trace(go.Scattergeo(
                                lat=_centroids['med_lat'],
                                lon=_centroids['med_lon'],
                                mode='markers',
                                marker=dict(symbol=_sym,
                                            size=int(_g_overlay_size),
                                            color=_cent_colors,
                                            line=dict(color='black', width=1.4)),
                                customdata=np.c_[_centroids['Group_Name'].astype(str),
                                                 _centroids['n']],
                                hovertemplate=((f'<b>%{{customdata[0]}}</b><br>'
                                                f'{_stat} centroid<br>'
                                                'Lat=%{lat:.2f}, Lon=%{lon:.2f}<br>'
                                                'N=%{customdata[1]}<extra></extra>')
                                               if _ov_hover_on else None),
                                hoverinfo=('skip' if not _ov_hover_on else None),
                                name=f'{_stat} centroid',
                                showlegend=False,
                            ))
                _mfig.update_layout(height=440, template='plotly_white',
                                     margin=dict(l=10, r=10, t=20, b=10),
                                     geo=dict(projection_type='natural earth'),
                                     dragmode='lasso')
                try:
                    _map_event = st.plotly_chart(_mfig, width='stretch',
                                                  key='g_map_lasso',
                                                  on_select='rerun',
                                                  selection_mode=['lasso', 'box'])
                except TypeError:
                    _map_event = None
                    st.plotly_chart(_mfig, width='stretch', key='g_map_static')
                _map_pos_idx = plotly_selected_indices(_map_event) if _map_event else []

            # Cross-plot (single trace + optional overlay traces for stats).
            # Scattergl uses WebGL — orders of magnitude faster than the SVG
            # Scatter for >5K points. The marker.line border isn't supported
            # by Scattergl (silently ignored), so we only set it on the
            # overlay/median markers (regular Scatter) where it actually
            # renders.
            with _mc2:
                _gfig = go.Figure()
                _samples_hover_on = (_g_hover_mode == 'All')
                _gfig.add_trace(go.Scattergl(
                    x=_vd[_vis_x], y=_vd[_vis_y], mode='markers',
                    marker=dict(size=int(_g_marker_size), color=_row_colors),
                    customdata=_customdata,
                    hovertemplate=(_hover_tmpl if _samples_hover_on else None),
                    hoverinfo=('skip' if not _samples_hover_on else None),
                    name='samples',
                    showlegend=False,
                ))
                # Build overlay traces but DEFER adding them — we want them
                # painted last (on top of the rolling band / smoother lines
                # that get added below) so the medians stay visible.
                _overlay_traces = []
                if _vis_overlay and not _summary.empty:
                    for _stat in _vis_overlay:
                        _ov_x_col = f'{_vis_x}_median' if _vis_x != _g_value_default else 'Median'
                        _ov_y_col = f'{_vis_y}_median' if _vis_y != _g_value_default else 'Median'
                        if _ov_x_col not in _summary.columns or _ov_y_col not in _summary.columns:
                            continue
                        _ov_hover = '<br>'.join([
                            'Group=%{customdata[0]}',
                            f'{_stat} {_vis_y}=%{{y:.2f}}',
                            f'{_stat} {_vis_x}=%{{x:.2f}}',
                            'N=%{customdata[1]}',
                            'MAD=%{customdata[2]:.2f}',
                        ])
                        # Match each overlay marker to its group's colour
                        # so the user can read group identity directly off
                        # the diamond/star — and so changing a group's
                        # colour propagates to the overlay too.
                        _ov_colors = [
                            _color_map.get(str(g), '#111827')
                            for g in _summary['Group_Name']
                        ]
                        # Overlay hover follows the same rule as the map
                        # centroids: on for "All" and "Overlays only", off
                        # for "Off". Median/mean tooltip is the whole
                        # point of these markers, so we keep it by default.
                        _ov_hover_on = (_g_hover_mode != 'Off')
                        _overlay_traces.append(go.Scatter(
                            x=_summary[_ov_x_col], y=_summary[_ov_y_col], mode='markers',
                            marker=dict(symbol='diamond' if _stat == 'Median' else 'star',
                                        size=int(_g_overlay_size),
                                        color=_ov_colors,
                                        line=dict(color='black', width=1.4)),
                            name=f'{_stat} per group',
                            customdata=np.c_[_summary['Group_Name'].astype(str),
                                              _summary['N'], _summary['MAD']],
                            hovertemplate=((_ov_hover + '<extra></extra>')
                                           if _ov_hover_on else None),
                            hoverinfo=('skip' if not _ov_hover_on else None),
                        ))
                # Landing 3 — rolling-window smoother along the PCA axis
                if _vis_x == 'Along_Strike_km' and _poly_on and 'Along_Strike_km' in _vd.columns:
                    _roll_x = pd.to_numeric(_vd['Along_Strike_km'], errors='coerce')
                    _roll_y = pd.to_numeric(_vd[_vis_y], errors='coerce')
                    _rgrid, _rsmooth, _rlo, _rhi = _g_rolling_window_smoothed(
                        _roll_x.to_numpy(), _roll_y.to_numpy(),
                        window_km=_poly_window, stat=_poly_stat,
                    )
                    if _rgrid.size > 0:
                        _band_ok = np.isfinite(_rsmooth) & np.isfinite(_rlo) & np.isfinite(_rhi)
                        if _band_ok.any():
                            _stat_label = ('Median + IQR' if _poly_stat == 'median'
                                           else 'Mean ± 1 SD')
                            # Filled IQR / ±SD band
                            _bx = np.concatenate([_rgrid[_band_ok], _rgrid[_band_ok][::-1]])
                            _by = np.concatenate([_rhi[_band_ok], _rlo[_band_ok][::-1]])
                            _gfig.add_trace(go.Scatter(
                                x=_bx, y=_by, mode='lines',
                                fill='toself',
                                fillcolor='rgba(30,58,95,0.10)',
                                line=dict(color='rgba(0,0,0,0)'),
                                showlegend=False, hoverinfo='skip',
                                name='rolling_band',
                            ))
                            # Smoothed centre line
                            _gfig.add_trace(go.Scatter(
                                x=_rgrid[_band_ok], y=_rsmooth[_band_ok],
                                mode='lines',
                                line=dict(color='#1e3a5f', width=2.5),
                                name=f'{_stat_label} ({_poly_window} km)',
                                hovertemplate=(
                                    f'Along-strike=%{{x:.1f}} km<br>'
                                    f'{_vis_y} {_stat_label}=%{{y:.2f}}'
                                    '<extra></extra>'
                                ),
                            ))
                # Now finally append the overlay (median/mean) traces — by
                # adding them last they paint on top of every other trace
                # (samples + rolling band + smoother), so they're never
                # hidden by the IQR fill or the smoothed centre line.
                for _ov in _overlay_traces:
                    _gfig.add_trace(_ov)
                _gfig.update_layout(height=440, template='plotly_white',
                                     margin=dict(l=10, r=10, t=20, b=10),
                                     dragmode='lasso',
                                     xaxis_title=_vis_x, yaxis_title=_vis_y,
                                     legend=dict(orientation='h', y=-0.15))
                try:
                    _xplot_event = st.plotly_chart(_gfig, width='stretch',
                                                    key='g_xplot_lasso',
                                                    on_select='rerun',
                                                    selection_mode=['lasso', 'box'])
                except TypeError:
                    _xplot_event = None
                    st.plotly_chart(_gfig, width='stretch', key='g_xplot_static')
                # Cross-plot's selected points may come from the samples trace OR an overlay
                # trace — we only want the samples trace (curve_number=0).
                _xplot_pos_idx = []
                if _xplot_event:
                    _sel = _xplot_event.get('selection', {}) if isinstance(_xplot_event, dict) else getattr(_xplot_event, 'selection', {})
                    _pts = (_sel.get('points', []) if isinstance(_sel, dict) else getattr(_sel, 'points', [])) or []
                    for _p in _pts:
                        _curve = _p.get('curve_number', _p.get('curveNumber', 0)) if isinstance(_p, dict) else getattr(_p, 'curve_number', 0)
                        if _curve != 0:
                            continue
                        _idx_val = _p.get('point_index', _p.get('pointIndex', _p.get('point_number'))) if isinstance(_p, dict) else getattr(_p, 'point_index', None)
                        if _idx_val is not None:
                            try:
                                _xplot_pos_idx.append(int(_idx_val))
                            except Exception:
                                pass
                    _xplot_pos_idx = sorted(set(_xplot_pos_idx))

            # Map both plots' positional indices back to original bench-index.
            # Two channels are unioned:
            #   1. Per-point hits from the Plotly event (limited to the
            #      ~10 K rows actually drawn after display-downsampling).
            #   2. Geometric polygon test against the *full* _vis_df bench
            #      so points hidden by the downsample are still captured —
            #      the user expects "everything inside the lasso", not
            #      "only the dots that happened to be drawn".
            _selected_bench_idx = set()
            for _pos in _map_pos_idx:
                if 0 <= _pos < len(_vd_orig_index):
                    _selected_bench_idx.add(int(_vd_orig_index[_pos]))
            for _pos in _xplot_pos_idx:
                if 0 <= _pos < len(_vd_orig_index):
                    _selected_bench_idx.add(int(_vd_orig_index[_pos]))
            # Full-bench geometric capture (covers downsampled-out rows)
            try:
                _selected_bench_idx |= plotly_polygon_indices(
                    _map_event, _vis_df, 'Lon', 'Lat',
                )
                _selected_bench_idx |= plotly_polygon_indices(
                    _xplot_event, _vis_df, _vis_x, _vis_y,
                )
            except Exception:
                pass

            # ── Lasso group management panel ─────────────────────────────
            st.markdown('**Lasso groups** — drag a lasso or box on either chart, then assign')
            _lg_state = st.session_state[_lasso_state_key]
            _lg_names = [v.get('name', k) for k, v in _lg_state.items()]
            _lg_keys = list(_lg_state.keys())

            _ag1, _ag2, _ag3, _ag4 = st.columns([1.2, 1.4, 1.0, 1.0])
            _new_name = _ag1.text_input(
                f'Selected: {len(_selected_bench_idx)} samples',
                value='',
                placeholder='Name for new group (or leave blank)',
                key='g_lasso_new_name',
            )
            _add_to_choice = _ag2.selectbox(
                'Add to existing group',
                ['(none)'] + _lg_names,
                key='g_lasso_add_to',
            )
            if _ag3.button('+ New group', key='g_lasso_new', disabled=(len(_selected_bench_idx) == 0)):
                _next_n = len(_lg_state) + 1
                _new_id = f'lasso_{_next_n}'
                while _new_id in _lg_state:
                    _next_n += 1
                    _new_id = f'lasso_{_next_n}'
                # Convert palette → hex up-front. The Bold/Set2 Plotly
                # palettes return ``rgb(102, 194, 165)`` strings which
                # st.color_picker rejects with StreamlitAPIException.
                _lg_state[_new_id] = {
                    'name': _new_name.strip() or f'Lasso {_next_n}',
                    'color': _to_hex_color(_lasso_palette[(_next_n - 1) % len(_lasso_palette)]),
                    'members': set(_selected_bench_idx),
                }
                st.rerun()
            if _ag4.button('Add to group', key='g_lasso_add',
                            disabled=(len(_selected_bench_idx) == 0 or _add_to_choice == '(none)')):
                # Resolve name → id
                _target_id = next((k for k, v in _lg_state.items()
                                    if v.get('name', k) == _add_to_choice), None)
                if _target_id is not None:
                    _lg_state[_target_id]['members'] = (
                        _lg_state[_target_id].get('members', set()) | set(_selected_bench_idx)
                    )
                    st.rerun()

            # List existing lasso groups with rename / colour-pick / delete
            # The colour picker writes to BOTH the lasso entry's own color
            # AND _user_colour_overrides keyed by name, so the change shows
            # up everywhere _color_map is consulted (map sample dots, map
            # centroids, cross-plot dots, cross-plot overlays, axis lines).
            if _lg_state:
                for _li, (_lid, _ldef) in enumerate(list(_lg_state.items())):
                    _lr1, _lr2, _lr3, _lr4, _lr5 = st.columns([2.0, 0.7, 1.0, 0.6, 0.6])
                    _curr_name = _ldef.get('name', _lid)
                    _new_label = _lr1.text_input(f'Group {_li + 1} name',
                                                   value=_curr_name,
                                                   key=f'g_lasso_rename_{_lid}',
                                                   label_visibility='collapsed')
                    if _new_label != _curr_name:
                        # Migrate colour-override key when group is renamed
                        if _curr_name in _user_colour_overrides:
                            _user_colour_overrides[_new_label] = _user_colour_overrides.pop(_curr_name)
                        _ldef['name'] = _new_label
                    # Coerce the stored value to hex on every render —
                    # legacy state from before the rgb→hex fix may still
                    # carry an ``rgb(...)`` string and would crash the
                    # picker. _to_hex_color is a no-op for valid hex.
                    _stored_hex = _to_hex_color(_ldef.get('color', '#888888'))
                    _picked = _lr2.color_picker(
                        f'Colour {_li + 1}',
                        value=_stored_hex,
                        key=f'g_lasso_color_{_lid}',
                        label_visibility='collapsed',
                    )
                    if _picked != _stored_hex:
                        _ldef['color'] = _picked
                        _user_colour_overrides[_ldef.get('name', _lid)] = _picked
                        st.rerun()
                    _lr3.markdown(f"**{len(_ldef.get('members', set()))} samples**")
                    if _lr4.button('Empty', key=f'g_lasso_empty_{_lid}',
                                    help='Clear this group\'s members but keep the entry'):
                        _ldef['members'] = set()
                        st.rerun()
                    if _lr5.button('✕', key=f'g_lasso_del_{_lid}', help='Delete this lasso group'):
                        _lg_state.pop(_lid, None)
                        # Also drop any colour override tied to this group's
                        # name so a later group with the same name doesn't
                        # silently inherit the previous colour.
                        _user_colour_overrides.pop(_ldef.get('name', _lid), None)
                        st.rerun()
                if st.button('Clear all lasso groups', key='g_lasso_clear_all'):
                    st.session_state[_lasso_state_key] = {}
                    st.rerun()
            else:
                st.caption('No lasso groups yet. Drag a lasso/box on the map or cross-plot, then click "+ New group".')

            # Colour-override panel for filter-derived groups (anything in
            # view that isn't a lasso group). Lets the user re-skin
            # filter-based groups too — same colour picker pattern, same
            # _user_colour_overrides store.
            _lasso_names_set = {_v.get('name', _k) for _k, _v in _lg_state.items()}
            _filter_groups_in_view = [
                gn for gn in _grp_names_in_view if gn not in _lasso_names_set
            ]
            if _filter_groups_in_view:
                with st.expander(f'Group colours — filter groups ({len(_filter_groups_in_view)})',
                                  expanded=False):
                    for _fi, _fgname in enumerate(_filter_groups_in_view):
                        _fc1, _fc2, _fc3 = st.columns([2.4, 0.7, 0.6])
                        _fc1.markdown(f'`{_fgname}`')
                        # Same rgb→hex coercion as the lasso panel — palette
                        # colours from Set2/Bold come through as rgb(...)
                        # strings that st.color_picker won't accept.
                        _stored_filter_hex = _to_hex_color(_color_map.get(_fgname, '#888888'))
                        _fpick = _fc2.color_picker(
                            f'Colour for {_fgname}',
                            value=_stored_filter_hex,
                            key=f'g_filter_color_{_g_src}_{_fi}',
                            label_visibility='collapsed',
                        )
                        if _fpick != _stored_filter_hex:
                            _user_colour_overrides[_fgname] = _fpick
                            st.rerun()
                        if _fgname in _user_colour_overrides:
                            if _fc3.button('Reset', key=f'g_filter_color_reset_{_g_src}_{_fi}',
                                           help='Drop the override and fall back to palette colour'):
                                _user_colour_overrides.pop(_fgname, None)
                                st.rerun()

    # Only show the save UI when at least one group is defined.
    if _n_passing > 0 and _n_groups >= 1:
        st.divider()
        _enabled_grouped = _grouped.dropna(subset=['Group_ID'])

        # ── Apply groups → write Group_ID back into the source dataset ────────
        # The grouping itself is the only artefact this tab produces.  Group
        # averaging (one-row-per-group) is *not* done here — it lives in the
        # Model / Validate / Predict tabs as a "sample data vs. group averages"
        # toggle, applied at use-time so you can switch without re-grouping.
        st.markdown('**Apply groups to dataset**')
        st.caption(
            f'Writes a `Group_ID` (and `Group_Name`) column directly into **{_g_src}** '
            'in the pool — no new dataset is created. Disabled groups are kept '
            'but lose their Group_ID label. Once applied, the Model, Validate, '
            'and Predict tabs will see the grouped dataset and offer a '
            '"sample data vs. group averages" toggle.'
        )
        if st.button(f'✓ Apply groups → {_g_src}', key='g_apply_groups',
                     use_container_width=True,
                     help='Writes Group_ID back into the source dataset in the pool. '
                          'Use the granularity toggle in Model/Validate/Predict to choose '
                          'whether to train/predict on raw samples or group medians.'):
            _pool = get_prepped_pool()
            # Build the source dataframe with Group_ID / Group_Name attached.
            # When _g_src is a built-in friendly name (e.g. 'Guo & Yang (2023)'),
            # save under the matching FILE STEM (e.g. 'GuoYang_2023_Model') so
            # downstream lookups via read_training_source / get_pool_for_tab
            # find it. Otherwise the grouped dataset would only be reachable
            # under '<friendly name> [grouped]' which the Model tab doesn't
            # check, and Multi-model training would crash with KeyError 'Group_ID'.
            if _g_src in _pool:
                # User-uploaded pool entry — modify in place under same key
                _src_df = _pool[_g_src].copy()
                _id_map = _grouped.set_index(_grouped.index)[['Group_ID', 'Group_Name']] \
                                  if 'Group_Name' in _grouped.columns \
                                  else _grouped[['Group_ID']].copy()
                for _col in ['Group_ID', 'Group_Name']:
                    if _col in _id_map.columns:
                        _src_df[_col] = _id_map[_col]
                _pool[_g_src] = _src_df.reset_index(drop=True)
                _saved_under = _g_src
            elif _g_src in _BUILTIN_TO_FILE_STEM:
                # Built-in reference — save under the file stem so
                # read_training_source picks it up for ML training.
                _stem = _BUILTIN_TO_FILE_STEM[_g_src]
                _src_df = load_dataset_by_name(_g_src).copy()
                if not _src_df.empty:
                    _src_df = _src_df.reset_index(drop=True)
                    # Align grouping by row position (both share the same bench origin)
                    if len(_src_df) == len(_grouped):
                        for _col in ['Group_ID', 'Group_Name']:
                            if _col in _grouped.columns:
                                _src_df[_col] = _grouped[_col].values
                    _pool[_stem] = _src_df
                    _saved_under = _stem
                else:
                    # Fallback: at least save the grouped slice directly
                    _pool[_stem] = _enabled_grouped.reset_index(drop=True)
                    _saved_under = _stem
            else:
                # Custom reference or unknown source — save with [grouped] suffix
                _saved_under = f'{_g_src} [grouped]'
                _pool[_saved_under] = _enabled_grouped.reset_index(drop=True)
            st.session_state['prepped_datasets'] = _pool
            st.session_state['_g_apply_preview'] = _enabled_grouped[
                ['Group_ID'] + [c for c in _enabled_grouped.columns if c != 'Group_ID'][:4]
            ].head(10)
            st.session_state['_g_apply_saved_under'] = _saved_under
            st.rerun()

        if st.session_state.get('_g_apply_preview') is not None:
            _saved_label = st.session_state.get('_g_apply_saved_under', _g_src)
            if _saved_label != _g_src:
                st.success(f'Group_ID written to **{_g_src}** (saved in pool as `{_saved_label}` — '
                           'Model / Validate / Predict will pick this up automatically).')
            else:
                st.success(f'Group_ID written to **{_g_src}**.')
            st.dataframe(st.session_state['_g_apply_preview'], hide_index=True, use_container_width=True)

    # --- Publish grouping map for Result-Summary tab ------------------------
    if _n_passing > 0:
        _gmap = _grouped.dropna(subset=['Group_ID']).copy()
        _gmap['Grouping_Method'] = 'Filter+Lasso'
        st.session_state['_rs_group_map'] = _gmap

with t_result_summary:
    st.header('Summary')
    _rs_pred=st.session_state.get('_rs_pred_bench',pd.DataFrame())
    _rs_group=st.session_state.get('_rs_group_map',pd.DataFrame())
    _rs_from_val = False
    if _rs_pred.empty:
        _rs_val_src = st.session_state.get('_rs_val_bench', pd.DataFrame())
        if not _rs_val_src.empty:
            _rs_pred = _rs_val_src.copy()
            _rs_from_val = True
    if _rs_pred.empty:
        st.info(
            'Run **Predict** (Predict tab) or **Validate** (Validate tab) first — '
            'either bench populates the Summary.'
        )
    else:
        if _rs_from_val:
            st.caption(
                '📋 Showing **validation** results — no prediction run yet. '
                'Run the Predict tab for full unknown-sample summaries.'
            )
        _rs_has_groups=not _rs_group.empty and 'Group_ID' in _rs_group and 'Grouping_Method' in _rs_group
        # ── Row 1: data / model controls ─────────────────────────────────────
        _rs_c1,_rs_c2,_rs_c3,_rs_c4=st.columns([1.3,1.2,1.1,0.9])
        if _rs_has_groups:
            _rs_methods=sorted(_rs_group['Grouping_Method'].dropna().astype(str).unique())
            _rs_method=_rs_c1.selectbox('Grouping method',_rs_methods,index=0,key='rs_violin_method')
            _rs_vdf=_rs_group[_rs_group['Grouping_Method'].astype(str)==_rs_method].copy()
            _rs_x='Group_ID'
        else:
            _rs_vdf=_rs_pred.copy()
            _rs_x='Model'
        if {'Lat','Lon'}.issubset(_rs_vdf) and _rs_x in _rs_vdf:
            _rs_vdf=attach_long_axis_position(_rs_vdf,_rs_x,'Long_Axis_Position_km')
        _rs_model_opts=sorted(_rs_vdf['Model'].dropna().astype(str).unique()) if 'Model' in _rs_vdf else []
        _rs_models=_rs_c2.multiselect('Models',_rs_model_opts,default=_rs_model_opts,key='rs_model_filter') if len(_rs_model_opts)>1 else _rs_model_opts
        if _rs_models and 'Model' in _rs_vdf:
            _rs_vdf=_rs_vdf[_rs_vdf['Model'].astype(str).isin(_rs_models)]
        # When the user has just defined groups but predictions haven't been
        # run on those grouped rows yet, _rs_group_map has Group_ID but no
        # Predicted_km. Render an info banner and bail rather than crashing
        # with KeyError.
        if 'Predicted_km' not in _rs_vdf.columns:
            st.info(
                'These groups don\'t have predicted thicknesses yet. '
                'Run predictions in the **Predict** tab on the same grouped '
                'dataset, then come back here for the violin / map / table.'
            )
            st.stop()
        _rs_vdf['Predicted_km']=pd.to_numeric(_rs_vdf['Predicted_km'],errors='coerce')
        _rs_vdf=_rs_vdf.dropna(subset=['Predicted_km',_rs_x]).copy()
        _rs_color_cat=[c for c in [_rs_x,'Grouping_Method','Model','Algorithm','Tectonic_Setting','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset'] if c in _rs_vdf and c!=_rs_x or c==_rs_x]
        _rs_color_cat=list(dict.fromkeys([_rs_x]+[c for c in _rs_color_cat if c!=_rs_x]))
        _rs_color_cont=[c for c in ['Predicted_km','Lat','Lon','Long_Axis_Position_km','Age_Ma'] if c in _rs_vdf]
        _rs_color_all=list(dict.fromkeys(_rs_color_cat+_rs_color_cont))
        _rs_clabels={**{c:local_option_label(c) for c in _rs_color_all},'Long_Axis_Position_km':'Along-strike [Km]','Group_ID':'Group','Predicted_km':'Predicted thickness [Km]'}
        _rs_color=_rs_c3.selectbox('Point colour by',_rs_color_all,index=0,key='rs_violin_color',format_func=lambda c: _rs_clabels.get(c,c))
        _rs_order_map={'Group name':_rs_x,'Median thickness':'Predicted_km','Latitude':'Lat','Longitude':'Lon'}
        if 'Long_Axis_Position_km' in _rs_vdf: _rs_order_map['Along-strike']='Long_Axis_Position_km'
        _rs_order_label=_rs_c4.selectbox('Order by',list(_rs_order_map),index=0,key='rs_violin_order')
        _rs_order_col=_rs_order_map[_rs_order_label]
        _rs_point_size=int(st.session_state.get('global_point_size', 5))
        # ── Row 2: violin appearance + overlay toggles ────────────────────────
        _rs_d1,_rs_d2,_rs_d3,_rs_d4,_rs_d5=st.columns([1.4,1.1,1.1,1.1,1.3])
        _rs_viol_colour=_rs_d1.selectbox('Violin colour',['By group','By median thickness'],key='rs_viol_colour')
        _rs_show_stats=_rs_d2.checkbox('Mean/median labels',value=True,key='rs_show_stats')
        _rs_show_n=_rs_d3.checkbox('Show N',value=True,key='rs_show_n')
        _rs_show_pts=_rs_d4.checkbox('Show points',value=True,key='rs_show_pts')
        # Compute group order
        if _rs_order_col==_rs_x or _rs_order_col not in _rs_vdf:
            _rs_group_order=sorted(_rs_vdf[_rs_x].dropna().astype(str).unique())
        else:
            _rs_ord_num=pd.to_numeric(_rs_vdf[_rs_order_col],errors='coerce')
            _rs_gmed=_rs_ord_num.groupby(_rs_vdf[_rs_x].astype(str)).median()
            _rs_group_order=_rs_gmed.sort_values().index.tolist()
        if _rs_vdf.empty:
            st.info('No data available after filtering.')
        else:
            _rs_color_is_num=(_rs_color in _rs_vdf
                and _rs_color not in (_rs_x,'Model','Algorithm','Grouping_Method','Tectonic_Setting','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset')
                and pd.api.types.is_numeric_dtype(pd.to_numeric(_rs_vdf[_rs_color],errors='coerce')))
            _rs_palette=qualitative_palette('Set2')
            # Violin fill colour helper
            _rs_thick_meds=_rs_vdf.groupby(_rs_vdf[_rs_x].astype(str))['Predicted_km'].median().to_dict()
            _thick_lo=min(_rs_thick_meds.values()); _thick_hi=max(_rs_thick_meds.values())
            def _viol_gc(grp,idx):
                if _rs_viol_colour=='By median thickness':
                    _t=(_rs_thick_meds.get(str(grp),_thick_lo)-_thick_lo)/max(_thick_hi-_thick_lo,1e-6)
                    return px.colors.sample_colorscale('plasma',[_t])[0]
                return _rs_palette[idx%len(_rs_palette)]
            # Global y-range for consistent point colorscale when coloring by Predicted_km
            _rs_y_min=_rs_vdf['Predicted_km'].min(); _rs_y_max=_rs_vdf['Predicted_km'].max()
            _rs_fig=go.Figure()
            for _gi,_grp in enumerate(_rs_group_order):
                _sub=_rs_vdf[_rs_vdf[_rs_x].astype(str)==str(_grp)]
                if _sub.empty: continue
                _gc=_viol_gc(_grp,_gi)
                _rs_fig.add_trace(go.Violin(
                    x=_sub[_rs_x].astype(str),y=_sub['Predicted_km'],name=str(_grp),
                    fillcolor=_gc,line_color=_gc,box_visible=True,meanline_visible=True,
                    opacity=0.65,showlegend=False,points=False,
                ))
                if not _rs_show_pts: continue
                if _rs_color_is_num:
                    _cvals=pd.to_numeric(_sub[_rs_color],errors='coerce')
                    # Fix colorscale disconnect: pin cmin/cmax to global y range when coloring by thickness
                    _cmin=_rs_y_min if _rs_color=='Predicted_km' else None
                    _cmax=_rs_y_max if _rs_color=='Predicted_km' else None
                    _rs_fig.add_trace(go.Scatter(
                        x=_sub[_rs_x].astype(str),y=_sub['Predicted_km'],mode='markers',
                        name=str(_grp),showlegend=False,
                        marker=dict(size=_rs_point_size,color=_cvals,colorscale='Viridis',
                                    cmin=_cmin,cmax=_cmax,
                                    showscale=(_gi==0),colorbar=dict(title=_rs_clabels.get(_rs_color,_rs_color),len=0.5,thickness=12),
                                    line=dict(color='black',width=0.3),opacity=0.7),
                        hovertemplate=f'{_grp}<br>H=%{{y:.1f}} km<br>{_rs_clabels.get(_rs_color,_rs_color)}=%{{marker.color:.2f}}<extra></extra>',
                    ))
                else:
                    _cat_color=_gc if _rs_color==_rs_x or _rs_color not in _sub else None
                    if _cat_color is None:
                        _cat_vals=_sub[_rs_color].fillna('unknown').astype(str)
                        _alt_pal=qualitative_palette('Plotly')
                        for _ci2,_cv2 in enumerate(list(dict.fromkeys(_cat_vals))):
                            _csub=_sub[_cat_vals==_cv2]
                            _rs_fig.add_trace(go.Scatter(
                                x=_csub[_rs_x].astype(str),y=_csub['Predicted_km'],mode='markers',name=f'{_cv2}',
                                showlegend=(_gi==0),legendgroup=_cv2,
                                marker=dict(size=_rs_point_size,color=_alt_pal[_ci2%len(_alt_pal)],opacity=0.6,line=dict(color='black',width=0.3)),
                                hovertemplate=f'{_grp}<br>H=%{{y:.1f}} km<br>{_rs_clabels.get(_rs_color,_rs_color)}={_cv2}<extra></extra>',
                            ))
                    else:
                        _rs_fig.add_trace(go.Scatter(
                            x=_sub[_rs_x].astype(str),y=_sub['Predicted_km'],mode='markers',name=str(_grp),
                            showlegend=False,marker=dict(size=_rs_point_size,color=_gc,opacity=0.55,line=dict(color='black',width=0.3)),
                            hovertemplate=f'{_grp}<br>H=%{{y:.1f}} km<extra></extra>',
                        ))
            # ── Mean/median value labels ──────────────────────────────────────
            if _rs_show_stats:
                for _grp in _rs_group_order:
                    _sub=_rs_vdf[_rs_vdf[_rs_x].astype(str)==str(_grp)]
                    if _sub.empty: continue
                    _mn=_sub['Predicted_km'].mean(); _md=_sub['Predicted_km'].median()
                    _rs_fig.add_trace(go.Scatter(
                        x=[str(_grp),str(_grp)],y=[_mn,_md],mode='text',
                        text=[f'<b>μ {_mn:.1f}</b>',f'M {_md:.1f}'],
                        textposition='middle right',showlegend=False,hoverinfo='skip',
                        textfont=dict(size=9,color='#1f2937'),
                    ))
            # ── N per group annotations ───────────────────────────────────────
            _rs_y_ann=_rs_y_min-4
            if _rs_show_n:
                for _grp in _rs_group_order:
                    _sub=_rs_vdf[_rs_vdf[_rs_x].astype(str)==str(_grp)]
                    if _sub.empty: continue
                    _n_grp=_sub['Sample_ID'].nunique() if 'Sample_ID' in _sub else len(_sub)
                    _rs_fig.add_annotation(
                        x=str(_grp),y=_rs_y_ann,text=f'n={_n_grp}',
                        showarrow=False,font=dict(size=9,color='#6b7280'),
                        xref='x',yref='y',yanchor='top',
                    )
            _rs_fig.update_layout(
                template='plotly_white',height=520,violingap=0.08,violinmode='overlay',
                margin=dict(l=20,r=20,t=20,b=50),
                showlegend=_rs_color not in (_rs_x,'Predicted_km','Lat','Lon','Long_Axis_Position_km','Age_Ma') and not _rs_color_is_num,
                xaxis_title=_rs_clabels.get(_rs_x,_rs_x),
                yaxis_title='Predicted crustal thickness [km]',
                yaxis=dict(range=[_rs_y_ann-3,_rs_y_max+5] if _rs_show_n else None),
            )
            if (len(_rs_models) if _rs_models else len(_rs_model_opts))>1:
                st.caption('Multiple models overlaid — filter to one model for a clean per-group view.')
            _rs_pcol,_rs_ecol=st.columns([6,1])
            with _rs_pcol:
                st.plotly_chart(_rs_fig,width='stretch',key='rs_violin_fig')
            with _rs_ecol:
                st.write('')
                try:
                    _rs_png=_rs_fig.to_image(format='png',width=1600,height=600,scale=2)
                    st.download_button('📷 PNG',_rs_png,'violin_plot.png','image/png',key='rs_png')
                except Exception:
                    st.caption('PNG: pip install kaleido')
            # ── Summary table ─────────────────────────────────────────────────
            # Dedupe the group-by list — when there are no Group_ID groups,
            # _rs_x is already 'Model' and we'd otherwise pass ['Model',
            # 'Model', 'Algorithm'] to groupby, which crashes reset_index.
            _rs_grp_cols=list(dict.fromkeys(
                c for c in [_rs_x,'Model','Algorithm'] if c in _rs_vdf
            ))
            _rs_sum=_rs_vdf.groupby(_rs_grp_cols,dropna=False).agg(
                N=('Predicted_km','count'),
                Median_km=('Predicted_km','median'),
                Mean_km=('Predicted_km','mean'),
                SD_km=('Predicted_km','std'),
                Q25_km=('Predicted_km',lambda x: x.quantile(0.25)),
                Q75_km=('Predicted_km',lambda x: x.quantile(0.75)),
            ).reset_index()
            # (downloads consolidated into the Export results section below)
            table_action_card('Group prediction summary',tidy_numbers(_rs_sum),'result_summary.csv','rs_group_summary_table')
            # ── Summary map ───────────────────────────────────────────────────
            # Same lazy-execution pattern as the Training sample map: the
            # Summary tab body re-executes on every interaction in Group /
            # Validate / Predict (Streamlit re-runs every tab body), so a
            # Mapbox scatter that touches the full prediction bench would
            # rebuild even when the user is working elsewhere. Default OFF
            # so users coming here just to export don't pay the cost; tick
            # to view.
            if {'Lat','Lon'}.issubset(_rs_vdf.columns):
                st.divider()
                st.subheader('Prediction map')
                if not st.checkbox(
                    'Render prediction map',
                    value=False,
                    key='rs_map_render',
                    help='Off by default — the map redraws on every Streamlit '
                         'rerun (including widget interactions in other tabs). '
                         'Tick to compute and display.',
                ):
                    st.caption('🗺️ Tick **Render prediction map** above to compute and display the map.')
                else:
                    _rs_mc1,_rs_mc2=st.columns([4,1])
                    _rs_map_colour_opts = [c for c in ['Predicted_km','Predicted_CI90_Width_km',_rs_x,'Age_Ma','Model'] if c in _rs_vdf]
                    _rs_map_col=_rs_mc2.selectbox('Map colour', _rs_map_colour_opts,
                        format_func=lambda c: 'Prediction uncertainty [90% CI width, Km]' if c == 'Predicted_CI90_Width_km' else c,
                        key='rs_map_col')
                    _rs_map_num=pd.api.types.is_numeric_dtype(pd.to_numeric(_rs_vdf[_rs_map_col],errors='coerce'))
                    _rs_map_cscale = 'RdYlGn_r' if _rs_map_col == 'Predicted_CI90_Width_km' else ('plasma' if _rs_map_num else None)
                    _rs_mapdf=_rs_vdf.dropna(subset=['Lat','Lon']).copy()
                    _rs_mapfig=px.scatter_map(
                        _rs_mapdf,lat='Lat',lon='Lon',color=_rs_map_col,
                        color_continuous_scale=_rs_map_cscale,
                        hover_name=_rs_x if _rs_x in _rs_mapdf.columns else None,
                        hover_data={c:':.1f' if _rs_map_num else True
                                    for c in ['Predicted_km','Predicted_CI90_Width_km','Age_Ma','Model'] if c in _rs_mapdf and c!=_rs_map_col},
                        zoom=1,height=440,
                    )
                    _rs_mapfig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
                    with _rs_mc1:
                        st.plotly_chart(_rs_mapfig,use_container_width=True,key='rs_map_fig')
                    # CI width stats callout
                    if _rs_map_col == 'Predicted_CI90_Width_km' and 'Predicted_CI90_Width_km' in _rs_mapdf.columns:
                        _ci_s = pd.to_numeric(_rs_mapdf['Predicted_CI90_Width_km'], errors='coerce').dropna()
                        if not _ci_s.empty:
                            _ci_med = _ci_s.median(); _ci_p90 = _ci_s.quantile(0.9)
                            _ci_high_n = (_ci_s > _ci_p90).sum()
                            st.caption(f'Median 90% CI width: **{_ci_med:.1f} Km** · P90: {_ci_p90:.1f} Km · {_ci_high_n} samples above P90 (high uncertainty)')

    # ══════════════════════════════════════════════════════════════════════════
    # ── Export results ────────────────────────────────────────────────────────
    st.divider()
    st.subheader('Export results')

    # Data available for export
    _exp_pred  = st.session_state.get('_rs_pred_bench', pd.DataFrame())   # all predictions
    _exp_group = st.session_state.get('_rs_group_map',  pd.DataFrame())   # grouped predictions
    _exp_val   = st.session_state.get('_rs_val_bench',  pd.DataFrame())   # validation bench
    _exp_train = st.session_state.get('dp_training_df', pd.DataFrame())   # processed training input

    # Main predictions sheet: prefer grouped (has Group_ID) over flat predictions
    _exp_main  = _exp_group if not _exp_group.empty else _exp_pred

    with st.expander('📥 Export results', expanded=True):

        # ── 1. Sheet selection ─────────────────────────────────────────────────
        st.markdown('**Sheets to include**')
        _esh_c = st.columns(5)
        _esh_preds  = _esh_c[0].checkbox('Predictions',    value=not _exp_main.empty,
                                          disabled=_exp_main.empty, key='exp_sh_preds',
                                          help='Sample-level prediction results with CI and grouping')
        _esh_summ   = _esh_c[1].checkbox('Group summary',  value=not _exp_main.empty,
                                          disabled=_exp_main.empty, key='exp_sh_summ',
                                          help='N · Median · Mean · SD · Q25/Q75 per group and model')
        _esh_val    = _esh_c[2].checkbox('Validation',     value=not _exp_val.empty,
                                          disabled=_exp_val.empty,  key='exp_sh_val',
                                          help='Held-out test results with residuals and CRUST1 comparison')
        _esh_train  = _esh_c[3].checkbox('Input data',     value=False,
                                          disabled=_exp_train.empty, key='exp_sh_train',
                                          help='Processed / mapped input data from the Prepare tab')
        _esh_imp    = _esh_c[4].checkbox('Feature importance', value=False, key='exp_sh_imp',
                                          help='Relative feature importance for each trained model')

        st.divider()

        # ── 2. Column selection (applies to Predictions & Validation sheets) ──
        st.markdown('**Columns** — Predictions & Validation sheets')

        # Determine universe of columns across both sheets
        _exp_avail_all = list(dict.fromkeys(
            list(_exp_main.columns) + list(_exp_val.columns)
        ))

        # Feature-importance columns from trained models
        _exp_imp_df = st.session_state.get('_cached_importance_df', pd.DataFrame())
        _exp_used_features = (
            list(_exp_imp_df['Feature'].dropna().unique())
            if not _exp_imp_df.empty and 'Feature' in _exp_imp_df else []
        )
        _exp_proxy_cols = [c for c in _exp_avail_all if any(
            c.startswith(p) for p in ['H_','GAME','Proxy','proxy','Sr_Y_H','La_Yb'])]

        # Column groups intersected with what's actually available
        def _exp_grp(keys):
            return [c for k in keys for c in _EXPORT_COL_GROUPS.get(k, []) if c in _exp_avail_all]

        # Shortcut presets — stored in session state so multiselect seeded correctly
        _exp_sel_key = '_exp_col_sel'
        if _exp_sel_key not in st.session_state:
            st.session_state[_exp_sel_key] = _exp_avail_all

        _sq = st.columns(7)
        if _sq[0].button('All',         key='exp_qa', help='Include every available column'):
            st.session_state[_exp_sel_key] = _exp_avail_all; st.rerun()
        if _sq[1].button('Core',        key='exp_qb', help='Sample ID · location · age · predicted H · CI · group'):
            st.session_state[_exp_sel_key] = _exp_grp(['identity','location','age','prediction','grouping','validation']); st.rerun()
        if _sq[2].button('+ Metadata',  key='exp_qc', help='Add tectonic setting, rock type, domain, belt'):
            st.session_state[_exp_sel_key] = list(dict.fromkeys(
                st.session_state[_exp_sel_key] + _exp_grp(['class']))); st.rerun()
        if _sq[3].button('+ Elements',  key='exp_qd', help='Add all major, REE, and trace element columns that are populated'):
            _geochem = _exp_grp(['major','ree','trace'])
            st.session_state[_exp_sel_key] = list(dict.fromkeys(
                st.session_state[_exp_sel_key] + _geochem)); st.rerun()
        if _sq[4].button('+ Ratios',    key='exp_qe', help='Add all computed geochemical ratios'):
            st.session_state[_exp_sel_key] = list(dict.fromkeys(
                st.session_state[_exp_sel_key] + _exp_grp(['ratio']))); st.rerun()
        if _sq[5].button('+ Used only', key='exp_qf',
                         help='Add only elements / ratios that appear as model features'):
            _used = [c for c in _exp_used_features if c in _exp_avail_all]
            st.session_state[_exp_sel_key] = list(dict.fromkeys(
                st.session_state[_exp_sel_key] + _used)); st.rerun()
        if _sq[6].button('Reset',       key='exp_qg'):
            st.session_state.pop(_exp_sel_key, None); st.rerun()

        _exp_sel = st.multiselect(
            'Selected columns (drag to reorder)',
            options=_exp_avail_all,
            default=[c for c in st.session_state.get(_exp_sel_key, _exp_avail_all)
                     if c in _exp_avail_all],
            format_func=_export_label,
            key='exp_col_multiselect',
            help='Human-readable column names will be used in the exported file',
        )
        # Keep multiselect in sync with session state
        if _exp_sel != st.session_state.get(_exp_sel_key):
            st.session_state[_exp_sel_key] = _exp_sel

        st.divider()

        # ── 3. Build and download ─────────────────────────────────────────────
        _exp_any = any([_esh_preds, _esh_summ, _esh_val, _esh_train, _esh_imp])
        if not _exp_any:
            st.info('Tick at least one sheet above, then click Download.')
        else:
            _exp_sheets: dict = {}

            def _exp_filter_cols(df):
                return df[[c for c in _exp_sel if c in df.columns]]

            if _esh_preds and not _exp_main.empty:
                _exp_sheets['Predictions'] = _exp_filter_cols(_exp_main)

            if _esh_summ and not _exp_main.empty and 'Predicted_km' in _exp_main.columns:
                _grp_col = 'Group_ID' if 'Group_ID' in _exp_main else 'Model'
                # Dedupe — when Group_ID is missing, _grp_col is already
                # 'Model' and we'd otherwise pass ['Model','Model'] to
                # groupby, which crashes reset_index.
                _grp_by_cols = list(dict.fromkeys(
                    c for c in [_grp_col, 'Model'] if c in _exp_main.columns
                ))
                if _grp_by_cols:
                    _exp_grp_agg = (
                        _exp_main.groupby(_grp_by_cols, dropna=False)
                        ['Predicted_km'].agg(
                            N='count',
                            Median_km='median', Mean_km='mean', SD_km='std',
                            Q25_km=lambda x: x.quantile(0.25),
                            Q75_km=lambda x: x.quantile(0.75),
                        ).reset_index()
                    )
                    _exp_sheets['Group summary'] = _exp_grp_agg

            if _esh_val and not _exp_val.empty:
                _exp_sheets['Validation'] = _exp_filter_cols(_exp_val)

            if _esh_train and not _exp_train.empty:
                _exp_sheets['Input data'] = _exp_train[[
                    c for c in _exp_train.columns
                    if not c.startswith('_') and c not in ('Rock_Type_Text_Class','FeO_Source')
                ]]

            if _esh_imp and not _exp_imp_df.empty:
                _exp_sheets['Feature importance'] = _exp_imp_df

            try:
                _exp_xlsx = _build_export_excel(_exp_sheets)
                st.download_button(
                    f'Download Excel — {len(_exp_sheets)} sheet(s) · {sum(len(v) for v in _exp_sheets.values()):,} rows',
                    data=_exp_xlsx,
                    file_name='mohometer_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type='primary',
                    key='exp_dl_xlsx',
                )
            except Exception as _exp_e:
                st.error(f'Could not build Excel: {_exp_e}')
                st.info('Install openpyxl for Excel export: `pip install openpyxl`')

    st.divider()

    # ── Session save / restore ────────────────────────────────────────────────
    import pickle as _pickle
    with st.expander('Session save / restore', expanded=False):
        st.caption('Pickle all loaded data, mappings, and predictions — restore later to continue where you left off.')
        _sess_keys = ['dp_training_df', 'dp_validation_df', 'dp_prediction_df',
                      '_rs_pred_bench', '_rs_group_map', '_rs_val_bench']
        _sess_keys += [k for k in st.session_state if k.startswith('dp_map_') or
                       k.startswith('dp_conf_') or k.startswith('dp_use_')]
        _save_payload: dict = {k: st.session_state[k] for k in _sess_keys if k in st.session_state}
        if _save_payload:
            try:
                st.download_button(f'Save session ({len(_save_payload)} items)',
                    data=_pickle.dumps(_save_payload), file_name='mohometer_session.pkl',
                    mime='application/octet-stream', key='dl_session_save')
            except Exception as _pe:
                st.warning(f'Could not serialise session: {_pe}')
        else:
            st.info('Nothing to save yet — load some data first.')
        _sess_restore = st.file_uploader('Restore session (.pkl)', type=['pkl'], key='dl_session_restore')
        if _sess_restore is not None:
            try:
                _loaded = _pickle.loads(_sess_restore.read())
                if isinstance(_loaded, dict):
                    for _rk, _rv in _loaded.items():
                        st.session_state[_rk] = _rv
                    st.success(f'Session restored — {len(_loaded)} items loaded.')
                else:
                    st.error('Not a valid Mohometer session file.')
            except Exception as _re:
                st.error(f'Could not restore session: {_re}')

    # ── Model export / import ─────────────────────────────────────────────────
    _pickle2 = _pickle
    with st.expander('Model export / import', expanded=False):
        st.caption('Export trained models as .pkl files; re-import to predict without retraining.')
        if models:
            for _mname, _mdata in models.items():
                try:
                    _mbytes = _pickle2.dumps({'name': _mname, 'model': _mdata['model'],
                                              'features': _mdata.get('features', []),
                                              'target': _mdata.get('target', 'Crust_Thickness')})
                    st.download_button(f'Export — {_mname}', data=_mbytes,
                        file_name=f'model_{re.sub(r"[^a-z0-9]+","_",_mname.lower())}.pkl',
                        mime='application/octet-stream',
                        key=f'dl_model_{re.sub(r"[^a-z0-9]+","_",_mname.lower())}')
                except Exception:
                    st.caption(f'Could not export {_mname}')
        else:
            st.info('Train a model in the Model tab first.')
        _model_restore = st.file_uploader('Import model (.pkl)', type=['pkl'], key='dl_model_restore')
        if _model_restore is not None:
            try:
                _mr = _pickle2.loads(_model_restore.read())
                if isinstance(_mr, dict) and 'model' in _mr and 'name' in _mr:
                    st.session_state.setdefault('_imported_models', {})[_mr['name']] = _mr
                    st.success(f"Model '{_mr['name']}' imported.")
                else:
                    st.error('Not a valid exported model file.')
            except Exception as _mre:
                st.error(f'Could not import model: {_mre}')