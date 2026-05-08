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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
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

APP_VERSION = 'v10.1 feature-set'
DEFAULT_CRUST1_GRID = Path('CRUST_1_0_excel.csv')
GAME_CALIBRATION_FILE = Path('LuffiDucea_2022_Calibration.csv')
DEFAULT_LITHOREF18_XYZ = Path('LithoRef18.xyz')
DEFAULT_LITHOREF18_ZIP = Path('Alfonso 2019 supplemental_files.zip')
_ONEDRIVE_LITHOREF18_ZIP = Path(r'C:\Users\david.holder\OneDrive - Barrick Gold Corporation\Documents - Global Exploration\Non Technical data\Training & Reference\ASC\4_2026 Manuals\Crustal Thickness\Papers\Alfonso 2019 supplemental_files.zip')
GAME_ALPHA_DEFAULT = 6.79
GAME_BETA_DEFAULT = 26.40
RESIDUAL_COLORSCALE = [[0.0,'#244575'], [0.48,'#d9e7f0'], [0.50,'#f7f7f4'], [0.52,'#f4dfcf'], [1.0,'#8f1729']]
SUMMARY_MEDIAN_COLOR = '#69c58e'
SUMMARY_MEAN_COLOR = '#f0b36a'
GUO_FEATURES = ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Sr','Y','Rb','Ba','Hf','Nb','Ta','Th']
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
CHON = {'La':0.237,'Sm':0.153,'Eu':0.058,'Gd':0.2055,'Yb':0.161}
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
    typed_value = ui.number_input(
        f'{label} value',
        min_value=min_value,
        max_value=max_value,
        value=slider_value,
        step=step,
        key=f'{key}_typed',
        disabled=disabled,
        label_visibility='collapsed',
    )
    if isinstance(value, int) and not isinstance(value, bool):
        return int(typed_value)
    return float(typed_value)

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
def read_table(file_or_path, guo_no_header=False, expected=None):
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
# Registry: (display_label, internal_name, category, conv)
# conv = None | ('elem_oxide', factor) | ('ppb', 1e-3) | ('utm', None)
_DP_REGISTRY = [
    # Metadata
    ('Sample ID',              'Sample_ID',          'meta',     None),
    ('Dataset',                'Dataset',            'meta',     None),
    ('Reference',              'Reference',          'meta',     None),
    ('Notes / Comments',       'Notes',              'meta',     None),
    ('Analytical Total [wt%]', 'Analytical_Total',   'meta',     None),
    # Location
    ('Lat [DD]',               'Lat',                'location', None),
    ('Lon [DD]',               'Lon',                'location', None),
    ('UTM Easting [m]',        'UTM_Easting',        'location', ('utm', None)),
    ('UTM Northing [m]',       'UTM_Northing',       'location', ('utm', None)),
    ('Elevation [km]',         'Elevation_km',       'location', None),
    ('Lat min [DD]',           'Lat_Min',            'location', None),
    ('Lat max [DD]',           'Lat_Max',            'location', None),
    ('Lon min [DD]',           'Lon_Min',            'location', None),
    ('Lon max [DD]',           'Lon_Max',            'location', None),
    # Age
    ('Age [Ma]',               'Age_Ma',             'age',      None),
    ('Age error [Ma]',         'Age_Err_Ma',         'age',      None),
    ('Age min [Ma]',           'Age_Min_Ma',         'age',      None),
    ('Age max [Ma]',           'Age_Max_Ma',         'age',      None),
    ('Era',                    'Geologic_Era',       'age',      None),
    ('Period',                 'Geologic_Period',    'age',      None),
    ('Epoch',                  'Geologic_Epoch',     'age',      None),
    ('Stage',                  'Geologic_Stage',     'age',      None),
    # Classification / Grouping
    ('Tectonic Setting',       'Tectonic_Setting',   'class',    None),
    ('Arc',                    'Arc',                'class',    None),
    ('Segment',                'Segment',            'class',    None),
    ('Domain',                 'Geologic_Domain',    'class',    None),
    ('Belt',                   'Belt',               'class',    None),
    ('Sub-belt',               'Sub_Belt',           'class',    None),
    ('Location',               'Location',           'class',    None),
    ('Lithology',              'Lithology_Type',     'class',    None),
    ('Lithology Grouping',     'Lithology_Grouping', 'class',    None),
    ('Country',                'Country',            'class',    None),
    # Target
    ('Crustal thickness [km]', 'Crust_Thickness',    'target',   None),
    # Major oxides and elements [wt%]
    ('SiO2 [wt%]',   'SiO2',    'major', None),
    ('Si [wt%]',     'SiO2',    'major', ('elem_oxide', 2.1394)),
    ('TiO2 [wt%]',   'TiO2',    'major', None),
    ('Ti [wt%]',     'TiO2',    'major', ('elem_oxide', 1.6685)),
    ('Al2O3 [wt%]',  'Al2O3',   'major', None),
    ('Al [wt%]',     'Al2O3',   'major', ('elem_oxide', 1.8895)),
    ('FeOt [wt%]',   'FeO',     'major', None),
    ('FeO [wt%]',    'FeO',     'major', None),
    ('Fe2O3t [wt%]', 'Fe2O3T',  'major', None),
    ('Fe2O3 [wt%]',  'Fe2O3',   'major', None),
    ('Fe [wt%]',     'FeO',     'major', ('elem_oxide', 1.2865)),
    ('MnO [wt%]',    'MnO',     'major', None),
    ('Mn [wt%]',     'MnO',     'major', ('elem_oxide', 1.2912)),
    ('MgO [wt%]',    'MgO',     'major', None),
    ('Mg [wt%]',     'MgO',     'major', ('elem_oxide', 1.6589)),
    ('CaO [wt%]',    'CaO',     'major', None),
    ('Ca [wt%]',     'CaO',     'major', ('elem_oxide', 1.3992)),
    ('Na2O [wt%]',   'Na2O',    'major', None),
    ('Na [wt%]',     'Na2O',    'major', ('elem_oxide', 1.3479)),
    ('K2O [wt%]',    'K2O',     'major', None),
    ('K [wt%]',      'K2O',     'major', ('elem_oxide', 1.2047)),
    ('P2O5 [wt%]',   'P2O5',    'major', None),
    ('P [wt%]',      'P2O5',    'major', ('elem_oxide', 2.2915)),
    ('LOI [wt%]',    'LOI',     'major', None),
    ('H2Ot [wt%]',   'H2Ot',    'major', None),
    # REE [ppm] and [ppb]
    ('La [ppm]','La','ree',None),  ('La [ppb]','La','ree',('ppb',1e-3)),
    ('Ce [ppm]','Ce','ree',None),  ('Ce [ppb]','Ce','ree',('ppb',1e-3)),
    ('Pr [ppm]','Pr','ree',None),  ('Pr [ppb]','Pr','ree',('ppb',1e-3)),
    ('Nd [ppm]','Nd','ree',None),  ('Nd [ppb]','Nd','ree',('ppb',1e-3)),
    ('Sm [ppm]','Sm','ree',None),  ('Sm [ppb]','Sm','ree',('ppb',1e-3)),
    ('Eu [ppm]','Eu','ree',None),  ('Eu [ppb]','Eu','ree',('ppb',1e-3)),
    ('Gd [ppm]','Gd','ree',None),  ('Gd [ppb]','Gd','ree',('ppb',1e-3)),
    ('Tb [ppm]','Tb','ree',None),  ('Tb [ppb]','Tb','ree',('ppb',1e-3)),
    ('Dy [ppm]','Dy','ree',None),  ('Dy [ppb]','Dy','ree',('ppb',1e-3)),
    ('Ho [ppm]','Ho','ree',None),  ('Ho [ppb]','Ho','ree',('ppb',1e-3)),
    ('Er [ppm]','Er','ree',None),  ('Er [ppb]','Er','ree',('ppb',1e-3)),
    ('Tm [ppm]','Tm','ree',None),  ('Tm [ppb]','Tm','ree',('ppb',1e-3)),
    ('Yb [ppm]','Yb','ree',None),  ('Yb [ppb]','Yb','ree',('ppb',1e-3)),
    ('Lu [ppm]','Lu','ree',None),  ('Lu [ppb]','Lu','ree',('ppb',1e-3)),
    # HFSE / LIL / compatible trace [ppm] and [ppb]
    ('Rb [ppm]','Rb','trace',None), ('Rb [ppb]','Rb','trace',('ppb',1e-3)),
    ('Sr [ppm]','Sr','trace',None), ('Sr [ppb]','Sr','trace',('ppb',1e-3)),
    ('Y [ppm]', 'Y', 'trace',None), ('Y [ppb]', 'Y', 'trace',('ppb',1e-3)),
    ('Zr [ppm]','Zr','trace',None), ('Zr [ppb]','Zr','trace',('ppb',1e-3)),
    ('Nb [ppm]','Nb','trace',None), ('Nb [ppb]','Nb','trace',('ppb',1e-3)),
    ('Ba [ppm]','Ba','trace',None), ('Ba [ppb]','Ba','trace',('ppb',1e-3)),
    ('Hf [ppm]','Hf','trace',None), ('Hf [ppb]','Hf','trace',('ppb',1e-3)),
    ('Ta [ppm]','Ta','trace',None), ('Ta [ppb]','Ta','trace',('ppb',1e-3)),
    ('Pb [ppm]','Pb','trace',None), ('Pb [ppb]','Pb','trace',('ppb',1e-3)),
    ('Th [ppm]','Th','trace',None), ('Th [ppb]','Th','trace',('ppb',1e-3)),
    ('U [ppm]', 'U', 'trace',None), ('U [ppb]', 'U', 'trace',('ppb',1e-3)),
    ('Sc [ppm]','Sc','trace',None), ('Sc [ppb]','Sc','trace',('ppb',1e-3)),
    ('V [ppm]', 'V', 'trace',None), ('V [ppb]', 'V', 'trace',('ppb',1e-3)),
    ('Cr [ppm]','Cr','trace',None), ('Cr [ppb]','Cr','trace',('ppb',1e-3)),
    ('Co [ppm]','Co','trace',None), ('Co [ppb]','Co','trace',('ppb',1e-3)),
    ('Ni [ppm]','Ni','trace',None), ('Ni [ppb]','Ni','trace',('ppb',1e-3)),
    ('Cu [ppm]','Cu','trace',None), ('Cu [ppb]','Cu','trace',('ppb',1e-3)),
    ('Zn [ppm]','Zn','trace',None), ('Zn [ppb]','Zn','trace',('ppb',1e-3)),
    ('Ga [ppm]','Ga','trace',None), ('Ga [ppb]','Ga','trace',('ppb',1e-3)),
    ('Li [ppm]','Li','trace',None), ('Li [ppb]','Li','trace',('ppb',1e-3)),
    ('Be [ppm]','Be','trace',None), ('Be [ppb]','Be','trace',('ppb',1e-3)),
    ('Cs [ppm]','Cs','trace',None), ('Cs [ppb]','Cs','trace',('ppb',1e-3)),
    ('Bi [ppm]','Bi','trace',None), ('Bi [ppb]','Bi','trace',('ppb',1e-3)),
    ('Sn [ppm]','Sn','trace',None), ('Sn [ppb]','Sn','trace',('ppb',1e-3)),
    ('W [ppm]', 'W', 'trace',None), ('W [ppb]', 'W', 'trace',('ppb',1e-3)),
    ('Mo [ppm]','Mo','trace',None), ('Mo [ppb]','Mo','trace',('ppb',1e-3)),
    ('Ge [ppm]','Ge','trace',None), ('Ge [ppb]','Ge','trace',('ppb',1e-3)),
    ('Cd [ppm]','Cd','trace',None), ('Cd [ppb]','Cd','trace',('ppb',1e-3)),
    ('Ag [ppm]','Ag','trace',None), ('Ag [ppb]','Ag','trace',('ppb',1e-3)),
    ('As [ppm]','As','trace',None), ('As [ppb]','As','trace',('ppb',1e-3)),
    ('Sb [ppm]','Sb','trace',None), ('Sb [ppb]','Sb','trace',('ppb',1e-3)),
    ('Tl [ppm]','Tl','trace',None), ('Tl [ppb]','Tl','trace',('ppb',1e-3)),
]

# {display_label: (internal_name, conv)}
_DP_DISPLAY_TO_INTERNAL = {dl: (iname, conv) for dl, iname, _cat, conv in _DP_REGISTRY}
# {internal_name: display_label}  — prefers conv=None entries
_DP_INTERNAL_TO_DISPLAY: dict = {}
for _dl, _iname, _cat, _conv in _DP_REGISTRY:
    if _iname not in _DP_INTERNAL_TO_DISPLAY and _conv is None:
        _DP_INTERNAL_TO_DISPLAY[_iname] = _dl

_DP_KEEP_ORIGINAL = '— keep original —'
_DP_DISPLAY_LABELS = [_DP_KEEP_ORIGINAL] + [dl for dl, _, _, _ in _DP_REGISTRY]

# ── Export column labels: internal_name → human-readable header ───────────────
_EXPORT_COL_LABELS: dict = {
    **{iname: dl for dl, iname, _cat, _cv in _DP_REGISTRY},   # elements/registry
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
_DP_USER_ALIAS_FILE = Path('_dp_user_aliases.json')

def _dp_load_user_aliases() -> dict:
    try:
        if _DP_USER_ALIAS_FILE.exists():
            return json.loads(_DP_USER_ALIAS_FILE.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}

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


# Major oxides included in anhydrous normalisation (volatiles excluded)
_DP_MAJOR_OXIDE_ANHY = ['SiO2','TiO2','Al2O3','FeO','Fe2O3','Fe2O3T','MnO','MgO',
                         'CaO','Na2O','K2O','P2O5']


def _dp_anhydrous_recalc(df: pd.DataFrame) -> pd.DataFrame:
    """Renormalise major oxides to 100% anhydrous (LOI/H2Ot excluded from sum)."""
    out = df.copy()
    ox = [c for c in _DP_MAJOR_OXIDE_ANHY if c in out.columns]
    if not ox:
        return out
    num = out[ox].apply(pd.to_numeric, errors='coerce')
    total = num.sum(axis=1, min_count=len(ox) // 2)   # need at least half populated
    for c in ox:
        out[c] = np.where(total.notna() & (total > 0), (num[c] / total * 100).round(4), out[c])
    return out


def _dp_alteration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """CIA · AI · CCPI · ICV · K/Al · Na/Al and mobility flags for every row."""
    def _n(col):
        return pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(np.nan, index=df.index)

    al2o3 = _n('Al2O3'); cao = _n('CaO'); na2o = _n('Na2O'); k2o = _n('K2O')
    feo   = _n('FeO');   mgo = _n('MgO'); mno  = _n('MnO'); tio2 = _n('TiO2')
    p2o5  = _n('P2O5');  loi = _n('LOI')

    # Molar masses for molar-ratio indexes
    MW = {'Al2O3': 101.96, 'CaO': 56.08, 'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94}

    al_m  = al2o3 / MW['Al2O3']
    na_m  = na2o  / MW['Na2O']
    k_m   = k2o   / MW['K2O']
    # CaO* corrected for Ca in apatite (Nesbitt & Young 1982)
    cao_star = (cao - 3.33 * p2o5).clip(lower=0)
    ca_m  = cao_star / MW['CaO']

    out = pd.DataFrame(index=df.index)

    # CIA — Chemical Index of Alteration (Nesbitt & Young 1982) [molar, ×100]
    # Fresh granite ~50; CIA 60-80 = moderate; >80 = intense weathering
    cia_d = al_m + ca_m + na_m + k_m
    out['CIA'] = np.where(cia_d > 0, (al_m / cia_d * 100).round(1), np.nan)

    # AI — Alteration Index (Ishikawa et al. 1976) [wt%, ×100]
    # Fresh felsic ~20; >50 = sericitic/potassic; combined with CCPI → box plot
    ai_n = k2o + mgo;  ai_d = k2o + mgo + na2o + cao
    out['AI'] = np.where(ai_d > 0, (ai_n / ai_d * 100).round(1), np.nan)

    # CCPI — Chlorite-Carbonate-Pyrite Index (Large et al. 2001) [wt%, ×100]
    # High CCPI + low AI = chloritic; high CCPI + high AI = carbonate/potassic
    ccpi_n = mgo + feo;  ccpi_d = mgo + feo + na2o + k2o
    out['CCPI'] = np.where(ccpi_d > 0, (ccpi_n / ccpi_d * 100).round(1), np.nan)

    # ICV — Index of Chemical Variation (Cox et al. 1995) [wt% ratio]
    # >1 = chemically immature/fresh; <1 = mature or recycled
    icv_n = cao + na2o + k2o + feo + mgo + mno + tio2
    out['ICV'] = np.where(al2o3 > 0, (icv_n / al2o3).round(3), np.nan)

    # K/Al and Na/Al (molar) for element-mobility diagram
    out['K/Al']  = np.where(al_m > 0, (k_m  / al_m).round(3), np.nan)
    out['Na/Al'] = np.where(al_m > 0, (na_m / al_m).round(3), np.nan)

    # Analytical total (sum of all oxide columns present including volatiles)
    ox_all = [c for c in ['SiO2','TiO2','Al2O3','FeO','Fe2O3','Fe2O3T','MnO','MgO',
                           'CaO','Na2O','K2O','P2O5','LOI','H2Ot'] if c in df.columns]
    if ox_all:
        out['Analytical total'] = (
            df[ox_all].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=4).round(2)
        )
        out['Total deviation'] = (out['Analytical total'] - 100.0).round(2)

    # LOI column (for display)
    out['LOI [wt%]'] = loi.round(2)

    # Flags
    out['LOI > 2%']      = loi > 2.0
    if 'Total deviation' in out:
        out['Total ±2%'] = out['Total deviation'].abs() > 2.0

    flag_cols = [c for c in ['LOI > 2%', 'Total ±2%'] if c in out]
    out['Any flag'] = out[flag_cols].any(axis=1) if flag_cols else False

    # Alteration interpretation (brief text label)
    def _interp(row):
        cia = row.get('CIA'); ai = row.get('AI'); ccpi = row.get('CCPI'); icv = row.get('ICV')
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [cia, ai, ccpi]):
            return ''
        tags = []
        if cia > 80:   tags.append('intense weathering')
        elif cia > 65: tags.append('moderate weathering')
        if ai > 70 and ccpi < 35:   tags.append('sericitic')
        elif ai < 30 and ccpi > 70: tags.append('chloritic')
        elif ai > 55 and ccpi > 55: tags.append('potassic/carbonate')
        if icv is not None and not np.isnan(icv) and icv < 0.85: tags.append('mature/recycled')
        return ', '.join(tags) if tags else 'fresh'

    # Mobile elements flag
    def _mobile(row):
        cia = row.get('CIA'); ai = row.get('AI'); ccpi = row.get('CCPI')
        els = []
        if cia and not np.isnan(cia) and cia > 65: els += ['Ca', 'Na']
        if ai  and not np.isnan(ai)  and ai  > 55: els += ['Na', 'Ca']
        if ccpi and not np.isnan(ccpi) and ccpi > 65: els += ['Na', 'K']
        return ', '.join(dict.fromkeys(els)) if els else ''

    out['Alteration']      = out.apply(_interp,  axis=1)
    out['Mobile elements'] = out.apply(_mobile,  axis=1)
    return out


def _dp_anhydrous_panel(df: pd.DataFrame, key_prefix: str) -> None:
    """Render the anhydrous recalculation toggle + alteration assessment UI inside an expander."""
    _has_loi   = 'LOI'    in df.columns
    _has_major = any(c in df.columns for c in _DP_MAJOR_OXIDE_ANHY)
    _has_al    = 'Al2O3'  in df.columns

    if not _has_major:
        st.info('Map major element columns to enable anhydrous recalculation and alteration assessment.')
        return

    # ── Anhydrous toggle ────────────────────────────────────────────────────
    _anhy_key  = f'{key_prefix}_anhy'
    _ignore_key = f'{key_prefix}_ignore'
    _full_key  = f'{key_prefix}_full_open'

    _ac1, _ac2 = st.columns([2, 3])
    _ac1.checkbox(
        'Normalise to 100% anhydrous',
        key=_anhy_key,
        help='Renormalise major oxides to sum to 100%, removing the LOI contribution. '
             'Required for thermodynamic modelling. Only available when LOI is mapped.',
        disabled=not _has_loi,
    )
    if not _has_loi:
        _ac1.caption('Map **LOI** to enable anhydrous normalisation.')

    # ── Alteration metrics ──────────────────────────────────────────────────
    _alt = _dp_alteration_metrics(df)
    _n_tot       = len(_alt)
    _n_loi_flag  = int((_alt['LOI > 2%']).sum())          if 'LOI > 2%'  in _alt else 0
    _n_tot_flag  = int((_alt['Total ±2%']).sum())         if 'Total ±2%' in _alt else 0
    _n_flagged   = int(_alt['Any flag'].sum())             if 'Any flag'  in _alt else 0
    _n_ignored   = len(st.session_state.get(_ignore_key, set()))

    if _n_loi_flag > 0:
        st.warning(
            f'⚠ **Element mobility warning** — {_n_loi_flag} of {_n_tot} samples have LOI > 2 wt%. '
            'Ca, Na and LILE may be partially mobilised.')
    if _n_tot_flag > 0:
        st.warning(
            f'⚠ {_n_tot_flag} samples have analytical total deviation > 2 wt%. '
            'Check for missing oxides or transcription errors.')

    # Metrics row
    if _has_al:
        _m1, _m2, _m3, _m4, _m5, _m6 = st.columns(6)
        def _med(col):
            s = _alt[col].dropna() if col in _alt else pd.Series(dtype=float)
            return f'{s.median():.1f}' if len(s) else '–'
        _m1.metric('CIA median', _med('CIA'),
                   help='Chemical Index of Alteration (Nesbitt & Young 1982). '
                        'Fresh: ~50; moderate weathering: 60–80; intense: >80')
        _m2.metric('AI median',  _med('AI'),
                   help='Alteration Index (Ishikawa 1976). '
                        'Fresh felsic ~20; sericitic >60; chloritic: low AI + high CCPI')
        _m3.metric('CCPI median',_med('CCPI'),
                   help='Chlorite-Carbonate-Pyrite Index (Large 2001). '
                        'Unaltered <20; chloritised >70 with low AI')
        _icv_s = _alt['ICV'].dropna() if 'ICV' in _alt else pd.Series(dtype=float)
        _m4.metric('ICV median', f'{_icv_s.median():.2f}' if len(_icv_s) else '–',
                   help='Index of Chemical Variation (Cox 1995). >1 fresh; <0.9 recycled/mature')
        _m5.metric('Flagged',    f'{_n_flagged}',
                   help='Samples with LOI > 2% or analytical total deviation > 2%')
        _m6.metric('Ignored',    f'{_n_ignored}',
                   help='Rows manually excluded from downstream processing')

    # K/Al vs Na/Al mobility diagram
    if 'K/Al' in _alt and 'Na/Al' in _alt:
        _kal = _alt[['K/Al','Na/Al','LOI [wt%]','Alteration']].dropna(subset=['K/Al','Na/Al'])
        if len(_kal) > 2:
            import plotly.express as _px_alt
            _kfig = _px_alt.scatter(
                _kal, x='Na/Al', y='K/Al', color='Alteration',
                hover_data={'LOI [wt%]': True},
                labels={'Na/Al': 'Na/Al (molar)', 'K/Al': 'K/Al (molar)'},
                template='plotly_white',
                color_discrete_sequence=['#3b82f6','#ef4444','#f59e0b','#10b981','#8b5cf6','#6b7280'],
            )
            _kfig.update_traces(marker=dict(size=5, opacity=0.7))
            _kfig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                legend=dict(orientation='h', y=-0.25))
            st.caption('K/Al vs Na/Al — element mobility diagram (fresh igneous: Na/Al ≈ 0.5, K/Al ≈ 0.2)')
            st.plotly_chart(_kfig, width='stretch', key=f'{key_prefix}_kal_fig')

    # ── Full sheet with flags ───────────────────────────────────────────────
    if st.button('View full sheet with flags', key=f'{key_prefix}_full_btn'):
        st.session_state[_full_key] = not st.session_state.get(_full_key, False)

    if st.session_state.get(_full_key, False):
        # Build display table: identity cols + metrics
        _id_cols = [c for c in ['Sample_ID','Lat','Lon'] if c in df.columns]
        _disp = pd.concat([df[_id_cols].reset_index(drop=True),
                           _alt.reset_index(drop=True)], axis=1)
        _ignore_set = st.session_state.get(_ignore_key, set())
        # Ignore column: True if Sample_ID in ignore set (or index-based fallback)
        if 'Sample_ID' in _disp.columns:
            _disp.insert(0, 'Ignore', _disp['Sample_ID'].astype(str).isin({str(s) for s in _ignore_set}))
        else:
            _disp.insert(0, 'Ignore', [i in _ignore_set for i in _disp.index])

        _locked = [c for c in _disp.columns if c != 'Ignore']
        _cc = {
            'Ignore': st.column_config.CheckboxColumn('Ignore row', default=False,
                help='Excluded from Model, Validate, and Predict tabs'),
            'LOI > 2%':   st.column_config.CheckboxColumn('LOI>2%',   disabled=True),
            'Total ±2%':  st.column_config.CheckboxColumn('Total±2%', disabled=True),
            'Any flag':   st.column_config.CheckboxColumn('Flagged',  disabled=True),
            'Alteration': st.column_config.TextColumn('Alteration', disabled=True, width='medium'),
            'Mobile elements': st.column_config.TextColumn('Mobile els', disabled=True),
        }
        for _nc in ['CIA','AI','CCPI','ICV','LOI [wt%]','Analytical total','Total deviation']:
            if _nc in _disp:
                _cc[_nc] = st.column_config.NumberColumn(_nc, disabled=True, format='%.1f')
        for _nc in ['K/Al','Na/Al']:
            if _nc in _disp:
                _cc[_nc] = st.column_config.NumberColumn(_nc, disabled=True, format='%.3f')

        _edited = st.data_editor(
            _disp, column_config=_cc, disabled=_locked,
            hide_index=True, use_container_width=True,
            key=f'{key_prefix}_full_editor', height=420,
        )
        # Persist ignore set
        if 'Sample_ID' in _edited.columns:
            _new_ignore = set(_edited.loc[_edited['Ignore'], 'Sample_ID'].astype(str))
        else:
            _new_ignore = set(_edited.index[_edited['Ignore']].tolist())
        if _new_ignore != _ignore_set:
            st.session_state[_ignore_key] = _new_ignore
            st.rerun()
        if _new_ignore:
            st.caption(f'{len(_new_ignore)} rows ignored and excluded from downstream tabs.')


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
    """Return sheet names for an Excel file without consuming the stream."""
    try:
        import openpyxl
        file.seek(0)
        wb = openpyxl.load_workbook(file, read_only=True, data_only=True)
        names = wb.sheetnames
        wb.close()
        file.seek(0)
        return names
    except Exception:
        try:
            file.seek(0)
            xl = pd.ExcelFile(file)
            names = xl.sheet_names
            file.seek(0)
            return names
        except Exception:
            return []


def _dp_read_raw(file, sheet_name=0):
    """Read file preserving original column names, with header-row detection.
    sheet_name: int index or str name (Excel only); ignored for CSV."""
    name = getattr(file, 'name', str(file)).lower()
    try:
        if name.endswith(('.xlsx', '.xls')):
            raw = pd.read_excel(file, header=None, sheet_name=sheet_name)
        else:
            raw = pd.read_csv(file, header=None)
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
    """Return {orig_col: display_label} — user-learned aliases > registry > keep-original."""
    user_aliases = _dp_load_user_aliases()
    result = {}
    for c in original_cols:
        k_norm = key(c)
        # User-corrected alias takes highest priority
        if k_norm in user_aliases and user_aliases[k_norm] in _DP_DISPLAY_TO_INTERNAL:
            result[c] = user_aliases[k_norm]
            continue
        is_ppb = ('ppb' in k_norm.split()) or k_norm.endswith('ppb') or '[ppb]' in str(c).lower()
        iname = canonical_header_name(c)
        if iname is None:
            result[c] = _DP_KEEP_ORIGINAL
            continue
        if is_ppb:
            ppb_dl = next((dl for dl, iname2, _cat, conv in _DP_REGISTRY
                           if iname2 == iname and conv is not None and conv[0] == 'ppb'), None)
            if ppb_dl:
                result[c] = ppb_dl
                continue
        result[c] = _DP_INTERNAL_TO_DISPLAY.get(iname, _DP_KEEP_ORIGINAL)
    return result


def _dp_apply_mapping(raw_df, mapping, utm_info=None):
    """Apply display_label mapping with unit conversions."""
    result = raw_df.copy()
    seen: dict = {}   # internal_name -> (orig_col, conv); first-mapped wins
    for orig_col, display_label in mapping.items():
        if orig_col not in result.columns or display_label == _DP_KEEP_ORIGINAL:
            continue
        if display_label not in _DP_DISPLAY_TO_INTERNAL:
            continue
        internal_name, conv = _DP_DISPLAY_TO_INTERNAL[display_label]
        if internal_name not in seen:
            seen[internal_name] = (orig_col, conv)
    for internal_name, (orig_col, conv) in seen.items():
        if conv is None or (conv and conv[0] == 'utm'):
            result = result.rename(columns={orig_col: internal_name})
        elif conv[0] in ('elem_oxide', 'ppb'):
            result[orig_col] = pd.to_numeric(result[orig_col], errors='coerce') * conv[1]
            result = result.rename(columns={orig_col: internal_name})
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
    # Compute geochemical ratios from element columns
    result = _dp_compute_ratios(result)
    if 'Sample_ID' not in result.columns:
        result.insert(0, 'Sample_ID', [f'Sample_{i+1}' for i in range(len(result))])
    return ensure_unique_columns(result)


_DP_LAT_LON_LABELS = {'Lat [DD]', 'Lon [DD]', 'Lat min [DD]', 'Lat max [DD]', 'Lon min [DD]', 'Lon max [DD]'}

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
    cols_present = [c for c in df.columns if c not in ('Sample_ID',)]
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

def find_training():
    for p in ['GuoYang_2023_Model.xlsx','Table S1(1).xlsx','Table S1.xlsx','CrustThickness_5Ma_Tibet_Normalized.csv','data/training/Table S1(1).xlsx']:
        if Path(p).exists(): return Path(p)
    return None

def find_zou_training():
    for p in ['Zou_2021_Model.xlsx','Zou_2021_Model.csv','Zou2021_Model.xlsx','Zou2021_Model.csv','Zou_2021.xlsx','Zou_2021.csv']:
        if Path(p).exists(): return Path(p)
    return None

def find_luffi_training():
    for p in ['LuffiDucea_2022_Calibration.csv','luffi_ducea_2022_game_calibration.csv','LuffiDucea_2022_Model.csv','LuffiDucea_2022_Model.xlsx']:
        if Path(p).exists(): return Path(p)
    return None

def _load_luffi_training(path):
    """Load the Luffi & Ducea (2022) calibration CSV as a training dataframe.
    Derives Crust_Thickness [km] from elevation using Moho = 6.79 × elev + 26.40."""
    df = read_table(path, guo_no_header=False)
    # Compute crustal thickness target from elevation if not already present
    if 'Crust_Thickness' not in df and 'Elevation_km' in df:
        elev = pd.to_numeric(df['Elevation_km'], errors='coerce')
        df['Crust_Thickness'] = (GAME_ALPHA_DEFAULT * elev + GAME_BETA_DEFAULT).round(2)
    # Drop rows without a usable target
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

def feature_importance(model, features):
    estimator = model.named_steps.get('model') if hasattr(model,'named_steps') else model
    if estimator is None or not hasattr(estimator,'feature_importances_'):
        return pd.DataFrame()
    imp = pd.DataFrame({'Feature':features,'Importance':estimator.feature_importances_})
    imp['Relative_Importance'] = imp['Importance'] / imp['Importance'].sum()
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
        fi = feature_importance(model,cfg['features'])
        if not fi.empty:
            fi.insert(0,'Model',label)
            fi.insert(1,'Feature_Set',cfg['feature_set'])
            fi.insert(2,'Algorithm',algorithm)
            importance.append(fi)
    val_df = pd.concat(validation,ignore_index=True) if validation else pd.DataFrame()
    imp_df = pd.concat(importance,ignore_index=True) if importance else pd.DataFrame()
    return models, val_df, imp_df

def read_training_source(source, uploaded=None):
    if source == 'From Data Prep tab':
        df = st.session_state.get('dp_training_df', pd.DataFrame())
        return df, 'Data Prep'
    if source == 'Upload dataset':
        if uploaded is None:
            return pd.DataFrame(), 'Uploaded dataset'
        return read_table(uploaded, guo_no_header=True, expected=TRAINING_COLUMNS), 'Uploaded dataset'
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
    # Guo & Yang (2023) default
    p = find_training()
    if p:
        return read_table(p, guo_no_header=True, expected=TRAINING_COLUMNS), 'Guo & Yang (2023)'
    return pd.DataFrame(), source

def dataset_numeric_features(df, target):
    skip = {'Sample_ID','Lat','Lon','Age_Ma',target,'Crust_Thickness'}
    return [c for c in df.columns if c not in skip and has_numeric_column(df, c)]

def preset_features(name, df, target):
    if name == 'Full suite':
        return dataset_numeric_features(df,target)
    features = FEATURE_SETS.get(name,FEATURE_SETS['Guo & Yang (2023)'])
    return [c for c in features if c in df]

def element_group(element):
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
    data = pd.DataFrame({'Feature':list(dict.fromkeys(features))})
    if data.empty:
        return data
    if importance is not None and not importance.empty and {'Feature','Relative_Importance'}.issubset(importance):
        imp = importance[['Feature','Relative_Importance']].copy()
        imp['Relative_Importance'] = pd.to_numeric(imp['Relative_Importance'], errors='coerce')
        imp = imp.groupby('Feature', as_index=False)['Relative_Importance'].max()
        data = data.merge(imp, on='Feature', how='left')
    else:
        data['Relative_Importance'] = np.nan
    data['Feature_Weighting_pct'] = pd.to_numeric(data['Relative_Importance'], errors='coerce') * 100.0
    if data['Feature_Weighting_pct'].notna().sum() == 0:
        data['Feature_Weighting_pct'] = 1.0
    data['Compatibility_Score'] = data['Feature'].map(compatibility_score)
    data['Behaviour'] = data['Compatibility_Score'].map(compatibility_label)
    return data

def feature_weighting_figure(features, importance=None, sort_by='Compatibility', height=260):
    data = feature_weighting_frame(features, importance)
    if data.empty:
        return None
    if sort_by == 'Importance':
        data = data.sort_values(['Feature_Weighting_pct','Compatibility_Score'], ascending=[False, True])
    else:
        data = data.sort_values(['Compatibility_Score','Feature_Weighting_pct'], ascending=[True, False])
    max_y = max(float(data['Feature_Weighting_pct'].max()) * 1.25, 1.0)
    fig = px.bar(
        data, x='Feature', y='Feature_Weighting_pct', color='Compatibility_Score',
        color_continuous_scale=[(0.0,'#253494'),(0.48,'#f4f4f5'),(0.52,'#f4f4f5'),(1.0,'#b91c1c')],
        range_color=[-1,1], template='plotly_white',
        labels={'Feature_Weighting_pct':'feature weighting (%)','Compatibility_Score':'compatible/incompatible'},
        custom_data=['Behaviour','Compatibility_Score']
    )
    fig.update_traces(marker_line_color='rgba(17,24,39,0.55)', marker_line_width=0.6, hovertemplate='<b>%{x}</b><br>weight=%{y:.1f}%<br>%{customdata[0]}<br>compatibility score=%{customdata[1]:+.1f}<extra></extra>')
    fig.add_hline(y=0, line=dict(color='rgba(17,24,39,0.55)', width=1))
    fig.add_annotation(x=0.01,y=1.08,xref='paper',yref='paper',text='more compatible',showarrow=False,font=dict(size=10,color='#253494'),xanchor='left')
    fig.add_annotation(x=0.99,y=1.08,xref='paper',yref='paper',text='more incompatible',showarrow=False,font=dict(size=10,color='#b91c1c'),xanchor='right')
    fig.update_layout(height=height,margin=dict(l=8,r=8,t=30,b=10),coloraxis_colorbar=dict(title='',len=0.72,y=0.48,thickness=10),showlegend=False)
    fig.update_xaxes(tickangle=-65,tickfont=dict(size=9),title='')
    fig.update_yaxes(range=[0,max_y],title='feature weighting (%)',tickfont=dict(size=9),title_font=dict(size=10),gridcolor='rgba(148,163,184,0.22)')
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
        'Other trace':'#f3f4f6',
    }
    for group in ['Major oxides','Transition metals','LILE','HFSE','REE','Other trace']:
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
            label = preset
        elif strategy == 'Minimum RI':
            min_ri_pct = typed_slider(st,f'{prefix} minimum full-model RI (%)',0.0,10.0,0.5,0.1,key=f'{prefix}_ri')
            features = threshold_feature_set(full_importance if full_importance is not None else pd.DataFrame(), min_ri_pct)
            label = f'RI >= {min_ri_pct:.1f}%'
            st.caption(f'{len(features)} features selected')
        else:
            numeric_options=dataset_numeric_features(df,target)
            features=st.multiselect(f'{prefix} custom elements',numeric_options,default=[c for c in preset_features(default_set,df,target) if c in numeric_options],key=f'{prefix}_custom')
            label='Custom'
        missing=[c for c in features if c not in df]
        if missing:
            st.warning(f'Missing columns: {", ".join(missing)}')
        render_element_summary([c for c in features if c in df],key=f'{prefix}_element_summary',full_importance=full_importance)
    return label, [c for c in features if c in df]

def training_subset_controls(prefix, df, expanded=False, target=None, noun='training'):
    if df.empty:
        return df
    filtered = enrich(df.copy(), la_mode)
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

def benchmark_figure(df, point_size=6, color_by='Model', show_fit=True, show_point_error=True, show_error_envelope=True, error_method='Window', window_km=10.0, min_n=10, use_age_window=False, age_window=10.0, show_tree_ci=False):
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
    if color_col == 'Delta_km':
        scatter_kwargs.update(color_continuous_scale='RdBu_r', color_continuous_midpoint=0)
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
            fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines+markers',name=f'{name} best fit',customdata=np.c_[delta,pct],line=dict(dash='dot'),hovertemplate='known: crustal thickness=%{x:.1f} Km<br>model best fit=%{y:.1f} Km<br>model - known=%{customdata[0]:+.1f} Km<br>%{customdata[1]:+.1f}%<extra></extra>'))
    fig.update_layout(legend=dict(groupclick='togglegroup'))
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
            fi = feature_importance(model,features)
            if not fi.empty:
                fi.insert(0,'Model',label)
                fi.insert(1,'Feature_Set',name)
                fi.insert(2,'Algorithm',algorithm)
                importance.append(fi)
    val_df = pd.concat(validation,ignore_index=True) if validation else pd.DataFrame()
    imp_df = pd.concat(importance,ignore_index=True) if importance else pd.DataFrame()
    return models, val_df, imp_df

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
    meta_cols = [c for c in ['Sample_ID','Lat','Lon','Age_Ma','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type','Rock_Type_Model','MgO','SiO2',target] + proxy_value_cols + proxy_thickness_cols + game_cols if c in test_df]
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
    return ensure_unique_columns(pd.concat(rows,ignore_index=True)) if rows else pd.DataFrame()

@st.cache_data(show_spinner=False)
def predict_uploaded(_models, pred_df, seed=42, la_yb_mode='raw_ppm'):
    rows = []
    pred_df = enrich(pred_df.copy(), la_yb_mode, include_game=True)
    proxy_thickness_cols = list(PROXY_THICKNESS_LABELS)
    proxy_value_cols = list(PROXY_VALUE_LIBRARY)
    game_cols = ['GAME_Luffi2022_N_mohometers','GAME_Luffi2022_N_raw_mohometers','GAME_Luffi2022_MAD_km','GAME_Luffi2022_IQR_km','GAME_Luffi2022_CI95_Low_km','GAME_Luffi2022_CI95_High_km','GAME_Luffi2022_CI95_Width_km','GAME_Luffi2022_Reliability','GAME_Luffi2022_Status']
    meta_cols = [c for c in ['Sample_ID','Lat','Lon','Age_Ma','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type','Rock_Type_Model','MgO','SiO2'] + proxy_value_cols + proxy_thickness_cols + game_cols if c in pred_df]
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
    return ensure_unique_columns(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()

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
    return '\n'.join(parts[:5]) if parts else 'All models appear ready, but no benchmark rows were produced.'

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

def game_calibration_figure(sample_df, sensor):
    cal = game_calibrators().get(sensor)
    if cal is None:
        return None
    labels = game_sensor_labels()
    raw_xy = cal['raw_xy']
    x_grid = np.linspace(float(np.nanmin(raw_xy[:,0])), float(np.nanmax(raw_xy[:,0])), 34)
    y_grid = np.linspace(float(np.nanmin(raw_xy[:,1])), float(np.nanmax(raw_xy[:,1])), 34)
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
        hovertemplate='MgO=%{x:.1f}<br>'+labels.get(sensor,sensor)+'=%{y:.1f}<br>surface Moho=%{z:.1f} km<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=raw_xy[:,0], y=raw_xy[:,1], mode='markers', name='T2 calibration points',
        marker=dict(size=6,color='rgba(255,255,255,0.85)',line=dict(color='black',width=0.7)),
        customdata=np.c_[cal['z'], GAME_ALPHA_DEFAULT * cal['z'] + GAME_BETA_DEFAULT],
        hovertemplate='MgO=%{x:.1f}<br>'+labels.get(sensor,sensor)+'=%{y:.1f}<br>calibration elevation=%{customdata[0]:.1f} km<br>calibration Moho=%{customdata[1]:.1f} km<extra></extra>'
    ))
    if sample_df is not None and not sample_df.empty and 'MgO' in sample_df:
        y = game_series(sample_df, sensor)
        overlay = pd.DataFrame({'MgO':pd.to_numeric(sample_df['MgO'], errors='coerce'), 'Mohometer':pd.to_numeric(y, errors='coerce')}).dropna()
        if not overlay.empty:
            fig.add_trace(go.Scatter(
                x=overlay['MgO'], y=overlay['Mohometer'], mode='markers', name='uploaded samples',
                marker=dict(size=8,color='rgba(230,80,70,0.90)',symbol='x'),
                hovertemplate='sample MgO=%{x:.1f}<br>'+labels.get(sensor,sensor)+'=%{y:.1f}<extra></extra>'
            ))
    fig.update_layout(template='plotly_white',height=460,margin=dict(l=20,r=20,t=20,b=20),legend=dict(orientation='h',y=-0.22,x=0))
    fig.update_xaxes(title='MgO [wt%]',showgrid=True)
    fig.update_yaxes(title=labels.get(sensor,sensor),showgrid=True)
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

def cluster_population_stats(values, target_value, min_cluster_n=5, seed=42):
    vals = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    empty = {
        'Thickness_Cluster_Count':np.nan,
        'Selected_Cluster_ID':np.nan,
        'Selected_Cluster_N':np.nan,
        'Selected_Cluster_Median':np.nan,
        'Selected_Cluster_Q25':np.nan,
        'Selected_Cluster_Q75':np.nan,
        'Selected_Cluster_Flag':'not run',
    }
    if len(vals) < max(10, 2 * int(min_cluster_n)) or pd.isna(target_value):
        return empty
    x = vals.to_numpy(dtype=float).reshape(-1,1)
    try:
        g1 = GaussianMixture(n_components=1, random_state=seed).fit(x)
        g2 = GaussianMixture(n_components=2, random_state=seed).fit(x)
        labels = g2.predict(x)
        counts = np.bincount(labels, minlength=2)
        if g2.bic(x) >= g1.bic(x) - 2 or counts.min() < int(min_cluster_n):
            empty['Thickness_Cluster_Count'] = 1
            empty['Selected_Cluster_Flag'] = 'single population'
            return empty
        means = g2.means_.ravel()
        selected_id = int(np.argmin(np.abs(means - float(target_value))))
        selected_vals = vals.iloc[np.where(labels == selected_id)[0]]
        q25 = float(selected_vals.quantile(0.25)); q75 = float(selected_vals.quantile(0.75))
        return {
            'Thickness_Cluster_Count':2,
            'Selected_Cluster_ID':selected_id + 1,
            'Selected_Cluster_N':int(len(selected_vals)),
            'Selected_Cluster_Median':float(selected_vals.median()),
            'Selected_Cluster_Q25':q25,
            'Selected_Cluster_Q75':q75,
            'Selected_Cluster_Flag':'two populations detected',
        }
    except Exception:
        return empty

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

def auto_labels_with_range(method, X, source, n_clusters, min_samples=8, max_group_range_km=np.nan,
                           max_h_range_km=np.nan, h_col='Predicted_km', max_clusters=12):
    method = method or 'KMeans'
    start_k = max(2, int(n_clusters))
    max_k = max(start_k, int(max_clusters))
    use_range = (pd.notna(max_group_range_km) and float(max_group_range_km) > 0) or (pd.notna(max_h_range_km) and float(max_h_range_km) > 0)
    methods_with_k = ['KMeans','Agglomerative']
    if method not in methods_with_k or not use_range:
        if method == 'Agglomerative':
            return AgglomerativeClustering(n_clusters=start_k).fit_predict(X), start_k, 'manual K'
        return KMeans(n_clusters=start_k, random_state=42, n_init=20).fit_predict(X), start_k, 'manual K'

    best_labels = None
    best_k = start_k
    best_reason = 'max K used; range target not fully met'
    for k in range(start_k, max_k + 1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X) if method == 'Agglomerative' else KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
        ok = True
        for label_id in np.unique(labels):
            g = source.iloc[np.where(labels == label_id)[0]]
            if len(g) < int(min_samples):
                ok = False
                break
            if pd.notna(max_group_range_km) and float(max_group_range_km) > 0 and {'Lat','Lon'}.issubset(g):
                axis = long_axis_km_from_lonlat(g['Lon'], g['Lat'])
                if pd.notna(axis) and axis > float(max_group_range_km):
                    ok = False
                    break
            if pd.notna(max_h_range_km) and float(max_h_range_km) > 0 and h_col in g:
                h = pd.to_numeric(g[h_col], errors='coerce').dropna()
                if len(h) > 1 and (h.max() - h.min()) > float(max_h_range_km):
                    ok = False
                    break
        best_labels = labels
        best_k = k
        if ok:
            best_reason = 'range target met'
            break
    return best_labels, best_k, best_reason

@st.cache_data(show_spinner=False)
def attach_auto_domains(df, feature_cols, method='KMeans', n_clusters=4, eps=0.85, min_samples=8,
                        spatial_weight=1.0, age_weight=1.0, thickness_weight=1.0, chemistry_weight=0.5,
                        label='Auto domain', auto_k_by_range=False, max_group_range_km=np.nan,
                        max_h_range_km=np.nan, h_col='Predicted_km', max_clusters=12):
    out = df.copy()
    out_col = label.replace(' ','_').replace('-','_') + '_ID'
    out[out_col] = pd.Series([pd.NA] * len(out), index=out.index, dtype='object')
    if out.empty or not {'Lat','Lon'}.issubset(out):
        return out, out_col, pd.DataFrame()
    cols = []
    weights = []
    if {'Lat','Lon'}.issubset(out):
        xy = pd.DataFrame(lonlat_xy_km(pd.to_numeric(out['Lon'],errors='coerce'), pd.to_numeric(out['Lat'],errors='coerce')), columns=['Auto_X_km','Auto_Y_km'], index=out.index)
        out = pd.concat([out,xy],axis=1)
        cols += ['Auto_X_km','Auto_Y_km']
        weights += [spatial_weight, spatial_weight]
    for c in feature_cols:
        if c in out and c not in cols:
            cols.append(c)
            if c == 'Age_Ma':
                weights.append(age_weight)
            elif c in ['Predicted_km','Observed_km','Residual_km'] or str(c).endswith('_km'):
                weights.append(thickness_weight)
            else:
                weights.append(chemistry_weight)
    work = out[cols].apply(pd.to_numeric, errors='coerce')
    keep_cols = [c for c in work.columns if work[c].notna().sum() >= max(3,int(min_samples))]
    work = work[keep_cols]
    weights = [w for c,w in zip(cols,weights) if c in keep_cols]
    if work.empty:
        return out, out_col, pd.DataFrame()
    ok = work.notna().any(axis=1)
    if ok.sum() < max(3,int(min_samples)):
        return out, out_col, pd.DataFrame()
    work = work.loc[ok].copy()
    for c in work.columns:
        med = work[c].median()
        work[c] = work[c].fillna(med)
    X = StandardScaler().fit_transform(work)
    X = X * np.asarray(weights, dtype=float)
    try:
        if method == 'DBSCAN':
            labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(X)
            chosen_k = int(len(set(labels)) - (1 if -1 in labels else 0))
            k_reason = 'DBSCAN eps/min samples'
        elif method == 'Agglomerative':
            labels, chosen_k, k_reason = auto_labels_with_range(
                'Agglomerative', X, out.loc[work.index], n_clusters, min_samples,
                max_group_range_km if auto_k_by_range else np.nan,
                max_h_range_km if auto_k_by_range else np.nan,
                h_col, max_clusters
            )
        else:
            labels, chosen_k, k_reason = auto_labels_with_range(
                'KMeans', X, out.loc[work.index], n_clusters, min_samples,
                max_group_range_km if auto_k_by_range else np.nan,
                max_h_range_km if auto_k_by_range else np.nan,
                h_col, max_clusters
            )
        out.loc[work.index,out_col] = [f'{label} {int(v)+1}' if int(v) >= 0 else f'{label} outlier' for v in labels]
    except Exception:
        return out, out_col, pd.DataFrame()
    summary = []
    for dom, g in out.dropna(subset=[out_col]).groupby(out_col, dropna=False):
        summary.append({
            'Grouping_Method':label,
            'Auto_Domain_ID':dom,
            'Chosen_K':chosen_k,
            'K_Selection':k_reason,
            'n':len(g),
            'Median_H_km':pd.to_numeric(g.get('Predicted_km'),errors='coerce').median() if 'Predicted_km' in g else np.nan,
            'Range_H_km':(pd.to_numeric(g.get(h_col),errors='coerce').max() - pd.to_numeric(g.get(h_col),errors='coerce').min()) if h_col in g else np.nan,
            'Median_Age_Ma':pd.to_numeric(g.get('Age_Ma'),errors='coerce').median() if 'Age_Ma' in g else np.nan,
            'Long_Axis_km':long_axis_km_from_lonlat(g['Lon'],g['Lat']) if {'Lon','Lat'}.issubset(g) and len(g.dropna(subset=['Lon','Lat'])) >= 2 else np.nan,
        })
    return out, out_col, tidy_numbers(pd.DataFrame(summary))

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
    out['Preset_Group_Segment_Long_Axis_km'] = np.nan
    for group_id, g in out.groupby('Preset_Group_ID', dropna=False):
        if {'Lon','Lat'}.issubset(g) and len(g.dropna(subset=['Lon','Lat'])) >= 2:
            axis = long_axis_km_from_lonlat(g['Lon'], g['Lat'])
        else:
            axis = max_segment_km if split_by_distance else np.nan
        if split_by_distance and pd.notna(axis):
            axis = min(float(axis), max_segment_km)
        out.loc[g.index, 'Preset_Group_Segment_Long_Axis_km'] = axis
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
                               rock_types=None, domain_col=None, cluster_diagnostic=False,
                               min_cluster_n=5, n_boot=500, seed=42, radius_col=None):
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
            if cluster_diagnostic:
                row.update(cluster_population_stats(selected[value_col] if value_col in selected else [], target.get(value_col), min_cluster_n=min_cluster_n, seed=seed+int(idx)))
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
        nearest = []
        for _, row in pts.iterrows():
            dist = haversine_km(float(row['Lat']), float(row['Lon']), glat, glon)
            nearest.append(int(np.nanargmin(dist)))
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
            assigned = False
            if domain_col == 'Geological_Domain':
                for domain_name, g in sample_coords.dropna(subset=[domain_col]).groupby(domain_col, dropna=False):
                    if len(g) >= 3 and polygon_contains_lonlat(np.column_stack([g['Lon'].to_numpy(dtype=float), g['Lat'].to_numpy(dtype=float)]), tlon, tlat):
                        out.at[idx, domain_col] = domain_name
                        assigned = True
                        break
            if not assigned:
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
            color=bg[value_col], colorscale='Viridis', opacity=opacity,
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

def style_training_map_legends(fig, color_by):
    title = model_map_legend_title(color_by)
    fig.update_layout(
        coloraxis_colorbar=dict(title=title,x=1.02,len=0.72,y=0.50),
        legend=dict(title=dict(text=title.replace('<br>',' ')),x=1.02,y=0.98,yanchor='top',bgcolor='rgba(255,255,255,0.78)')
    )
    return fig

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
    # Keep every Nth row — simple but preserves order/geography
    step = max(1, len(df) // limit)
    return df.iloc[::step].head(limit).copy()

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
pred_no_header=False
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
        st.info('Select at least one non-manual grouping method (Auto KMeans, Sample distribution long axis, etc.) to show the grouping map and proxy graph.')
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

t_data_prep,t0,t_validation,t_unknown,t_result_summary=st.tabs(['Prepare','Model','Validate','Predict','Summary'])

models={}; validation_df=pd.DataFrame(); importance_df=pd.DataFrame(); train_df=pd.DataFrame(); clean=pd.DataFrame(); target=None
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

    _dp_uploads = st.file_uploader(
        'Upload file(s)',
        type=['csv','xlsx','xls'],
        accept_multiple_files=True,
        key='dp_global_uploader',
        help='Upload any number of files — training, validation, prediction, or mixed. Assign each below.',
    )

    # Buckets filled per file; merged and stored at the end
    _dp_dest_dfs: dict = {'dp_training_df': [], 'dp_validation_df': [], 'dp_prediction_df': []}
    _dp_key = 'dp'   # single namespace for session-state keys

    for _dp_fi, _dp_f in enumerate(_dp_uploads or []):
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
                    # Sheet changed → force remap
                    _remap_key = f'{_dp_key}_map_{_dp_fi}_{_dp_f.name}'
                    st.session_state.pop(_remap_key, None)
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

        _dp_card_expanded = st.session_state.get(_dp_card_key, not _dp_all_confirmed)
        with st.expander(f'**File {_dp_fi+1} — {_dp_f.name}**' + (' ✓' if _dp_all_confirmed else ''), expanded=_dp_card_expanded):

            # Per-file session-state keys
            # Validate session state — reset if stale (old internal-name format)
            if (_dp_mkey not in st.session_state or
                    any(v not in _DP_DISPLAY_LABELS for v in st.session_state[_dp_mkey].values())):
                st.session_state[_dp_mkey] = _dp_auto_map(_dp_orig)
            if _dp_ckey not in st.session_state:
                st.session_state[_dp_ckey] = {c for c,m in st.session_state[_dp_mkey].items() if m == _DP_KEEP_ORIGINAL}
            _dp_cur_map   = st.session_state[_dp_mkey]
            _dp_confirmed = st.session_state[_dp_ckey]

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
                        st.session_state[_dp_ckey] = set()
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
                        st.session_state[_dp_ckey] = _xl_new_conf
                        st.success('Excel mapping imported.')
                        st.rerun()
                    except ValueError as _xe:
                        st.error(f'Could not read Excel mapping: {_xe}')

            with st.expander('Edit column mapping', expanded=(_dp_n_auto > 0)):
                _dp_filter_val = st.text_input(
                    '', key=f'{_dp_key}_filter_{_dp_fi}_{_dp_f.name}',
                    placeholder=_dp_search_ph, label_visibility='collapsed',
                    help='Click a header above to jump here, or type to filter rows',
                )
                _dp_filt_orig = [c for c in _dp_orig if not _dp_filter_val
                                 or _dp_filter_val.lower() in c.lower()
                                 or _dp_filter_val.lower() in _dp_cur_map.get(c,'').lower()]
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
                # Build duplicate-mapping warning: flag display labels used by >1 source col
                _dp_all_mapped = [
                    v for v in _dp_cur_map.values() if v != _DP_KEEP_ORIGINAL
                ]
                _dp_dup_labels = {v for v in _dp_all_mapped if _dp_all_mapped.count(v) > 1}
                if _dp_dup_labels:
                    _dup_pairs = []
                    for _dl in sorted(_dp_dup_labels):
                        _cols_with = [c for c, v in _dp_cur_map.items() if v == _dl]
                        _dup_pairs.append(f'**{_dl}** ← {", ".join(_cols_with)}')
                    st.warning('⚠ Same code mapped to multiple columns — only the last one will be used:\n\n' +
                               '\n\n'.join(_dup_pairs))

                _dp_map_df = pd.DataFrame({
                    'Original header': _dp_filt_orig,
                    'Map to':   [_dp_cur_map.get(c, _DP_KEEP_ORIGINAL) for c in _dp_filt_orig],
                    'Confirmed':[c in _dp_confirmed or _dp_cur_map.get(c, _DP_KEEP_ORIGINAL) == _DP_KEEP_ORIGINAL for c in _dp_filt_orig],
                    'Model importance': [_dp_imp_bar(_DP_DISPLAY_TO_INTERNAL.get(_dp_cur_map.get(c, ''), (c, None))[0]) for c in _dp_filt_orig],
                    '⚠': ['⚠ dup' if _dp_cur_map.get(c, _DP_KEEP_ORIGINAL) in _dp_dup_labels else '' for c in _dp_filt_orig],
                })
                _dp_edited = st.data_editor(
                    _dp_map_df,
                    column_config={
                        'Original header': st.column_config.TextColumn('Original header', disabled=True),
                        'Map to': st.column_config.SelectboxColumn('Map to', options=_DP_DISPLAY_LABELS, required=True),
                        'Confirmed': st.column_config.CheckboxColumn('Confirmed'),
                        'Model importance': st.column_config.TextColumn('Model importance', disabled=True,
                            help='Relative importance of this feature in the last trained model (blank = not a model feature or no model trained yet)'),
                        '⚠': st.column_config.TextColumn('⚠', disabled=True,
                            help='Duplicate: same registry code mapped to more than one column'),
                    },
                    hide_index=True, use_container_width=True,
                    key=f'{_dp_key}_editor_{_dp_fi}_{_dp_f.name}',
                )
                # Merge edited values back — only update the VISIBLE (filtered) rows,
                # preserving mappings/confirmations for columns hidden by the filter.
                _dp_filt_set   = set(_dp_filt_orig)
                _dp_edited_map = dict(zip(_dp_edited['Original header'], _dp_edited['Map to']))
                _dp_new_map    = {**_dp_cur_map, **_dp_edited_map}   # full map, filtered rows win
                _dp_edited_conf = set(_dp_edited.loc[_dp_edited['Confirmed'], 'Original header'])
                _dp_new_conf = (
                    {c for c in _dp_confirmed if c not in _dp_filt_set} | _dp_edited_conf
                )
                _dp_changed = (
                    any(_dp_edited_map.get(c) != _dp_cur_map.get(c) for c in _dp_filt_orig) or
                    any((c in _dp_new_conf) != (c in _dp_confirmed) for c in _dp_filt_orig)
                )
                if _dp_changed:
                    for _orig_c in _dp_filt_orig:
                        _nl = _dp_edited_map.get(_orig_c, _DP_KEEP_ORIGINAL)
                        if _nl != _dp_cur_map.get(_orig_c, _DP_KEEP_ORIGINAL):
                            _dp_save_user_alias(_orig_c, _nl)
                    st.session_state[_dp_mkey] = _dp_new_map
                    st.session_state[_dp_ckey] = _dp_new_conf
                    st.rerun()

                # ── Export / Edit-in-Excel ───────────────────────────────────
                _exp_c1, _exp_c2 = st.columns(2)
                _tpl_bytes = json.dumps(_dp_cur_map, indent=2, ensure_ascii=False).encode('utf-8')
                _exp_c1.download_button(
                    'Export mapping template (.json)',
                    data=_tpl_bytes,
                    file_name=f'mapping_template_{_dp_f.name}.json',
                    mime='application/json',
                    key=f'{_dp_key}_tpl_dl_{_dp_fi}_{_dp_f.name}',
                    help='Save mapping as a JSON template to re-use with similar files',
                )
                _xl_bytes = _dp_mapping_to_excel(_dp_orig, _dp_cur_map, _dp_confirmed)
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
            if 'UTM Easting [m]' in _dp_cur_map.values() and 'UTM Northing [m]' in _dp_cur_map.values():
                st.info('UTM coordinates detected — specify zone to convert to WGS84 DD.')
                _uc1, _uc2, _uc3 = st.columns([1,2,2])
                _dp_utm_zone = _uc1.number_input('UTM Zone', 1, 60, 30, 1, key=f'{_dp_key}_utm_zone_{_dp_fi}_{_dp_f.name}')
                _dp_utm_hemi = _uc2.radio('Hemisphere', ['Northern','Southern'], horizontal=True, key=f'{_dp_key}_utm_hemi_{_dp_fi}_{_dp_f.name}')
                _dp_utm_info = (int(_dp_utm_zone), _dp_utm_hemi == 'Northern')
                _uc3.caption(f'WGS84 {int(_dp_utm_zone)}{"N" if _dp_utm_info[1] else "S"} → DD')

            # Confirm / Reset buttons
            _ba, _bb = st.columns(2)
            if _ba.button('Confirm all', key=f'{_dp_key}_confirm_all_{_dp_fi}_{_dp_f.name}'):
                st.session_state[_dp_ckey] = set(_dp_orig)
                st.session_state[_dp_card_key] = False
                st.rerun()
            if _bb.button('Reset mapping', key=f'{_dp_key}_reset_{_dp_fi}_{_dp_f.name}'):
                st.session_state.pop(_dp_mkey, None); st.session_state.pop(_dp_ckey, None); st.rerun()

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

            # ── Processed DataFrame — cached by mapping hash to avoid re-running transforms ──
            import hashlib as _hl
            _dp_proc_key = f'{_dp_key}_proc_{_dp_fi}_{_dp_f.name}'
            _dp_map_hash = _hl.md5(json.dumps(sorted(_dp_cur_map.items()), ensure_ascii=False).encode()).hexdigest()
            _dp_utm_hash = str(_dp_utm_info)
            _dp_proc_sig  = f'{_dp_map_hash}_{_dp_utm_hash}'
            _dp_proc_sig_key = f'{_dp_proc_key}_sig'
            if (st.session_state.get(_dp_proc_sig_key) != _dp_proc_sig or
                    _dp_proc_key not in st.session_state):
                st.session_state[_dp_proc_key] = _dp_apply_mapping(_dp_raw, _dp_cur_map, _dp_utm_info)
                st.session_state[_dp_proc_sig_key] = _dp_proc_sig
            _dp_processed = st.session_state[_dp_proc_key]

            # ── Data Quality panel ───────────────────────────────────────────
            with st.expander('Data quality check', expanded=False):
                _dp_qa_panel(_dp_processed, key_prefix=f'{_dp_key}_{_dp_fi}_{_dp_f.name}')

            # ── Anhydrous recalculation & alteration panel ───────────────────
            with st.expander('Anhydrous recalculation & alteration', expanded=False):
                _dp_anhydrous_panel(_dp_processed, key_prefix=f'{_dp_key}_{_dp_fi}_{_dp_f.name}')

            # ── Build export df (anhydrous + ignore applied) ─────────────────
            _anhy_on_now = st.session_state.get(f'{_dp_key}_{_dp_fi}_{_dp_f.name}_anhy', False)
            _dp_display  = _dp_anhydrous_recalc(_dp_processed) if (_anhy_on_now and 'LOI' in _dp_processed.columns) else _dp_processed
            _dp_ignored  = st.session_state.get(f'{_dp_key}_{_dp_fi}_{_dp_f.name}_ignore', set())
            _dp_export_df = _dp_display.loc[~_dp_display.index.isin(_dp_ignored)].reset_index(drop=True)

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
                    _in, _cv = _DP_DISPLAY_TO_INTERNAL[_dl]
                    _tx = 'rename' if _cv is None else ('ppb → ppm (÷ 1000)' if _cv[0] == 'ppb' else (f'element → oxide (× {_cv[1]:.4f})' if _cv[0] == 'elem_oxide' else ('UTM → WGS84 DD' if _cv[0] == 'utm' else str(_cv))))
                    _prov_rows.append({'File': _pf.name, 'Original column': _oc, 'Mapped to': _in, 'Display label': _dl, 'Transformation': _tx})
        if _prov_rows:
            _prov_df = pd.DataFrame(_prov_rows)
            st.dataframe(_prov_df, hide_index=True, use_container_width=True)
            st.download_button('Download provenance CSV', _prov_df.to_csv(index=False).encode('utf-8'),
                               'column_provenance.csv', 'text/csv', key='dl_provenance')
        else:
            st.info('No mappings confirmed yet.')

with t0:
    st.header('Machine Learning Model')
    _dp_train_ready = not st.session_state.get('dp_training_df', pd.DataFrame()).empty
    source_options=['Guo & Yang (2023)','Zou et al. (2021)']
    if find_luffi_training(): source_options.append('Luffi & Ducea (2022)')
    source_options.append('Upload dataset')
    if _dp_train_ready: source_options.append('From Data Prep tab')
    primary_source=st.selectbox('1. Training model',source_options,index=0)
    primary_upload=None
    if primary_source == 'Upload dataset':
        primary_upload=st.file_uploader('Upload primary training dataset',type=['csv','xlsx','xls'],key='primary_training_dataset')
    elif primary_source == 'From Data Prep tab':
        st.info(f"Using {len(st.session_state['dp_training_df']):,} rows from Data Prep tab.")
    c1,c2=st.columns([1,2])
    try:
        train_df,source_label=read_training_source(primary_source,primary_upload)
        if not train_df.empty:
            train_df,target=target_column_controls('Primary model',train_df)
            if target is None:
                st.stop()
            primary_train_df=training_subset_controls('Primary model',train_df,expanded=False)
            default_set = ('Zou et al. (2021)' if primary_source == 'Zou et al. (2021)'
                           else 'Luffi & Ducea (2022)' if primary_source == 'Luffi & Ducea (2022)'
                           else 'Guo & Yang (2023)' if primary_source == 'Guo & Yang (2023)'
                           else 'Full suite')
            importance_features = FEATURE_SETS['Guo & Yang (2023)'] if all(c in primary_train_df for c in FEATURE_SETS['Guo & Yang (2023)']) else dataset_numeric_features(primary_train_df,target)
            full_model, full_clean = train_model(primary_train_df,target,importance_features,seed,'ExtraTrees')
            full_importance = feature_importance(full_model,importance_features)
            configs=[]
            with c1:
                default_algorithm = 'XGBoost' if primary_source == 'Zou et al. (2021)' and 'XGBoost' in ALGORITHMS else 'ExtraTrees'
                normalize_widget_state('primary_algorithm',ALGORITHMS)
                primary_algorithm=st.selectbox('Primary ML method',ALGORITHMS,index=ALGORITHMS.index(default_algorithm) if default_algorithm in ALGORITHMS else 0,key='primary_algorithm')
            with c2:
                m1,m2=st.columns(2)
                with m1:
                    summary_tile('Training source',source_label)
                with m2:
                    summary_tile('Training rows',len(full_clean),numeric=True)
            primary_feature_set,primary_features=feature_strategy_controls('Primary model',primary_train_df,target,default_set,full_importance,collapsed=False,allow_min_ri=False,expander_label='Primary element list')
            if primary_features:
                default_primary_name=f'Primary: {primary_feature_set} / {primary_algorithm}'
                normalize_widget_state('primary_model_name')
                primary_model_name=st.text_input('Primary model name',value=default_primary_name,key='primary_model_name')
                primary_model_name=display_algorithm_label(primary_model_name)
                configs.append({'label':primary_model_name.strip() or default_primary_name,'df':primary_train_df,'target':target,'features':primary_features,'feature_set':primary_feature_set,'algorithm':primary_algorithm})
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
                            configs.append({'label':model2_name.strip() or default_model2_name,'df':model2_df,'target':model2_target,'features':model2_features,'feature_set':model2_feature_set,'algorithm':model2_algorithm})
            models,validation_df,importance_df=train_configured_models(configs,seed)
            if not importance_df.empty:
                st.session_state['_cached_importance_df'] = importance_df.copy()
            primary_label=display_algorithm_label(configs[0]['label']) if configs else ''
            if primary_label:
                model=models[primary_label]['model']; clean=models[primary_label]['clean']; selected_features=models[primary_label]['features']; feature_set_name=primary_label
            with c2:
                summary_tile('Models',len(models),numeric=True,compact=True)
            st.divider()
            model_options=list(models.keys())
            normalize_widget_state('active_interp_model',model_options)
            active_model_label=st.selectbox('Model to use for interpretation',model_options,index=0,key='active_interp_model')
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
    except Exception as e:
        st.error(f'Model setup failed: {e}')

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
                cva,cvb,cvc,cvd=st.columns(4)
                cv_color=cva.selectbox('Colour points by',['Delta_km','Model','Algorithm'],index=0,key='model_cv_color',format_func=lambda c: 'model - known [Km]' if c == 'Delta_km' else c)
                cv_size=typed_slider(cvb,'Point size',2,12,5,1,key='model_cv_point_size')
                cv_fit=cvc.checkbox('Best-fit lines',True,key='model_cv_best_fit')
                cv_envelope=cvd.checkbox('Best-fit residual envelopes',False,key='model_cv_envelope')
                st.caption('R2 shows fit quality. RMSE and MAE are average prediction error in Km. Bias is mean model - known thickness; positive means the model overestimates.')
                st.dataframe(display_validation_summary(cv_plot),width='stretch')
                st.plotly_chart(benchmark_figure(tidy_numbers(cv_plot),cv_size,cv_color,cv_fit,False,cv_envelope,'Window',10.0,25,False,5.0),width='stretch')

    with st.expander('Feature importance',expanded=False):
        if importance_df.empty:
            st.info('Feature importance appears after the model is trained.')
        else:
            fi1,fi2=st.columns([1,0.7])
            normalize_widget_state('importance_model',list(models.keys()))
            model_for_importance=fi1.selectbox('Feature importance model',list(models.keys()),key='importance_model')
            importance_sort=fi2.selectbox('Sort features by',['Importance','Compatibility'],index=0,key='importance_sort')
            top=importance_df[importance_df['Model'].eq(model_for_importance)].sort_values('Relative_Importance',ascending=False).copy()
            fig=feature_weighting_figure(top['Feature'].tolist(),top,sort_by=importance_sort,height=360)
            if fig is not None:
                st.plotly_chart(fig,width='stretch')

    with st.expander('Training sample map',expanded=False):
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
                    labels={color_by:model_map_legend_title(color_by).replace('<br>',' ')}
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

with t_validation:
    st.header('Validation')
    st.caption('Use this tab to test trained models against known crustal-thickness data that were not used for training.')
    _dp_val_ready=not st.session_state.get('dp_validation_df',pd.DataFrame()).empty
    _val_src_opts=['Upload file']+(['From Data Prep tab'] if _dp_val_ready else [])
    _val_src=st.radio('Validation data source',_val_src_opts,horizontal=True,key='val_data_source')
    if _val_src=='From Data Prep tab':
        test_df=st.session_state['dp_validation_df'].copy()
        st.success(f"Using Data Prep validation data: {len(test_df):,} rows.")
        test_up=None
    else:
        test_up=st.file_uploader('Upload known validation dataset',type=['csv','xlsx','xls'],key='validation_known_upload')
        test_df=read_table(test_up,guo_no_header=True,expected=TRAINING_COLUMNS) if test_up else pd.DataFrame()
    if test_df.empty and _val_src=='Upload file' and not test_up:
        st.info('Upload blind validation data with chemistry plus a known crustal-thickness target. If Lat/Lon are present, CRUST1.0 can be used as the validation target.')
    elif not test_df.empty:
        if validation_df.empty:
            st.info('Train at least one model in the Model tab to benchmark against this data.')
        else:
            test_df,test_target=target_column_controls('Validation dataset',test_df)
            if test_target is None:
                st.warning('Choose a validation target column before running validation.')
            else:
                test_df=training_subset_controls('Validation dataset',test_df,expanded=True,target=test_target,noun='validation')
                v1,v2,v3,v4,v5,v6=st.columns(6)
                color_by=v1.selectbox('Colour points by',['Delta_km','Model','Algorithm'],index=0,key='validation_color',format_func=lambda c: 'model - known [Km]' if c == 'Delta_km' else c)
                point_size=typed_slider(v2,'Point size',2,12,5,1,key='validation_point_size')
                show_best_fit=v3.checkbox('Best-fit lines',True,key='validation_best_fit')
                show_point_error=v4.checkbox('Point residual bars',False,key='validation_point_error')
                show_envelope=v5.checkbox('Best-fit residual envelopes',False,key='validation_envelope')
                show_tree_ci=v6.checkbox('Tree 90% CI bars',False,key='validation_tree_ci',help='Show ensemble uncertainty: 90% interval across individual decision trees (ExtraTrees / RandomForest only)')
                test_bench=benchmark_uploaded(models,test_df,test_target,seed,la_mode)
                if not test_bench.empty and {'Lat','Lon'}.issubset(test_bench):
                    test_bench=attach_crust1_reference(test_bench)
                st.session_state['_rs_val_bench'] = test_bench.copy() if not test_bench.empty else pd.DataFrame()
                if test_bench.empty:
                    st.warning('No complete blind validation rows could be benchmarked for the selected models.')
                    report=benchmark_readiness_report(models,test_df,test_target)
                    if not report.empty:
                        st.error(benchmark_failure_message(report))
                        st.caption('Benchmark readiness by model')
                        table_action_card('Benchmark readiness by model',report,'benchmark_readiness.csv','validation_benchmark_readiness')
                else:
                    st.subheader('Blind validation graph')
                    summary=validation_summary(test_bench)
                    st.caption('R2 shows fit quality. RMSE and MAE are average prediction error in Km. Bias is mean model - known thickness; positive means the model overestimates.')
                    if show_tree_ci and 'Predicted_CI90_Low_km' not in test_bench:
                        st.info('Tree CI requires ExtraTrees or RandomForest — not available for boosting algorithms.')
                    table_action_card('Validation summary',display_validation_summary(test_bench),'validation_summary.csv','validation_summary')
                    fig=benchmark_figure(tidy_numbers(test_bench),point_size,color_by,show_best_fit,show_point_error,show_envelope,'Window',10.0,10,False,10.0,show_tree_ci)
                    st.plotly_chart(fig,width='stretch')

                    if 'H_GAME_LuffiDucea2022_km' in test_bench:
                        with st.expander('GAME diagnostics',expanded=False):
                            st.caption('Luffi & Ducea (2022) GAME mohometers are reconstructed from the published T2 calibration table using local LOWESS-style interpolation, then combined with the paper-style MAD filtering into a consensus Moho estimate.')
                            gtab1,gtab2,gtab3=st.tabs(['Consensus','Reliability','Calibration'])
                            with gtab1:
                                gx1,gx2=st.columns([1,0.35])
                                game_x=gx1.selectbox('Reference axis',['Observed_km','Predicted_km'],index=0,key='game_diag_x',format_func=lambda c: 'known: crustal thickness [Km]' if c == 'Observed_km' else 'model: crustal thickness [Km]')
                                game_point_size=typed_slider(gx2,'Point size',2,12,5,1,key='game_diag_point_size')
                                gfig=game_consensus_figure(test_bench,game_x,game_point_size)
                                if gfig is not None:
                                    st.plotly_chart(gfig,width='stretch',key='val_game_consensus_fig')
                                status_counts=test_bench['GAME_Luffi2022_Status'].fillna('unknown').value_counts().rename_axis('Status').reset_index(name='Rows') if 'GAME_Luffi2022_Status' in test_bench else pd.DataFrame()
                                if not status_counts.empty:
                                    table_action_card('GAME status counts',status_counts,'game_status_counts.csv','game_status_counts')
                            with gtab2:
                                st.caption('The original GAME app explicitly tracks how many mohometers survive data-availability, reference-model residual/RMSE, and STD/MAD filtering. Low kept N, high MAD/IQR, wide bootstrap CI, or an "all valid; high spread" status should be treated as lower-confidence interpretation.')
                                rx1,rx2=st.columns([1,0.35])
                                game_rel_x=rx1.selectbox('Reference for reliability residual',['Observed_km','Predicted_km'],index=0,key='game_reliability_x',format_func=lambda c: 'known: crustal thickness [Km]' if c == 'Observed_km' else 'model: crustal thickness [Km]')
                                game_rel_size=typed_slider(rx2,'Point size',2,12,6,1,key='game_reliability_point_size')
                                rfig=game_reliability_figure(test_bench,game_rel_x,game_rel_size)
                                if rfig is not None:
                                    st.plotly_chart(rfig,width='stretch',key='val_game_reliability_fig')
                                rel_counts=test_bench['GAME_Luffi2022_Reliability'].fillna('unknown').value_counts().rename_axis('Reliability').reset_index(name='Rows') if 'GAME_Luffi2022_Reliability' in test_bench else pd.DataFrame()
                                if not rel_counts.empty:
                                    table_action_card('GAME reliability counts',rel_counts,'game_reliability_counts.csv','game_reliability_counts')
                                rel_summary=game_reliability_summary(test_bench,game_rel_x)
                                if not rel_summary.empty:
                                    table_action_card('GAME reliability by N',rel_summary,'game_reliability_by_n.csv','game_reliability_by_n')
                            with gtab3:
                                sensor_options=[s for s,_,_ in GAME_SENSORS if s in game_calibrators()]
                                if sensor_options:
                                    sensor=st.selectbox('Mohometer calibration',sensor_options,index=0,key='game_calibration_sensor',format_func=lambda s: game_sensor_labels().get(s,s))
                                    cfig=game_calibration_figure(test_bench,sensor)
                                    if cfig is not None:
                                        st.plotly_chart(cfig,width='stretch',key='val_game_calibration_fig')
                                table_action_card('GAME mohometer summary',game_sensor_summary(),'game_mohometer_summary.csv','game_mohometer_summary')

                    st.subheader('Proxy comparison')
                    proxy_options=proxy_value_options_available(test_bench)
                    proxy_thickness_options=proxy_library_methods_available(test_bench)
                    if proxy_options or proxy_thickness_options:
                        x_options=['Observed_km'] + [f'Predicted_km::{m}' for m in test_bench['Model'].dropna().astype(str).unique()] + [f'Proxy::{p}' for p in proxy_thickness_options]
                        def proxy_x_label(option):
                            if option == 'Observed_km':
                                return 'known: crustal thickness [Km]'
                            if option.startswith('Predicted_km::'):
                                return model_thickness_label(option.split('::',1)[1])
                            if option.startswith('Proxy::'):
                                return PROXY_THICKNESS_LABELS.get(option.split('::',1)[1], option.split('::',1)[1]) + ' [Km]'
                            return option
                        color_labels={'Residual_km':'model - known [Km]','Age_Ma':'Age [Ma]','Rock_Type_Model':'Rock type','Tectonic_Setting':'Tectonic setting','Arc_or_Segment':'Arc/segment','Geologic_Domain':'Geological domain','Dataset':'Dataset'}
                        proxy_mode=st.radio('Proxy plot mode',['Proxy value','Proxy thickness'],horizontal=True,key='validation_proxy_mode')
                        pc1,pc2,pc3=st.columns([1,1,1])
                        normalize_widget_state('validation_proxy_x',x_options)
                        x_choice=pc1.selectbox('X axis', x_options, index=0, key='validation_proxy_x', format_func=proxy_x_label)
                        show_proxy_formulas=pc2.checkbox('Show formulas on graph',False,key='validation_proxy_show_formulas')
                        proxy_df=tidy_numbers(test_bench)
                        x_col='Observed_km'
                        if x_choice.startswith('Predicted_km::'):
                            model_choice=x_choice.split('::',1)[1]
                            proxy_df=proxy_df[proxy_df['Model'].astype(str).eq(model_choice)]
                            x_col='Predicted_km'
                        elif x_choice.startswith('Proxy::'):
                            x_col=x_choice.split('::',1)[1]
                        fig=None; _proxy_formulas={}
                        if proxy_mode == 'Proxy value':
                            if not proxy_options:
                                fig=None
                            else:
                                with st.expander('Proxy value controls',expanded=True):
                                    pcat1,pcat2,pcat3,pcat4=st.columns([1.4,1.1,1,0.45])
                                    proxy=pcat1.selectbox('Y axis',proxy_options,index=0,key='validation_proxy',format_func=proxy_value_label)
                                    proxy_color_options=[c for c in ['Residual_km','Age_Ma','Rock_Type_Model','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset'] if c in test_bench]
                                    proxy_color=pcat2.selectbox('Colour by',proxy_color_options,index=0,key='validation_proxy_color',format_func=lambda c: color_labels.get(c,c)) if proxy_color_options else ''
                                    proxy_point_size=typed_slider(pcat3,'Point size',2,12,5,1,key='validation_proxy_value_point_size')
                                    pt1,pt2,pt3=st.columns(3)
                                    proxy_trend=pt1.selectbox('Moving summary',['Median','Mean','Both','Off'],index=0,key='validation_proxy_value_trend')
                                    proxy_trend_bin=typed_slider(pt2,'Summary bin width [Km]',2.0,20.0,5.0,1.0,key='validation_proxy_value_trend_bin')
                                    proxy_trend_min_n=typed_slider(pt3,'Summary minimum N',1,50,5,1,key='validation_proxy_value_trend_min_n')
                                    _clip_opts={'None':0,'0.5%':0.5,'1%':1.0,'2%':2.0,'5%':5.0}
                                    _clip=_clip_opts[st.selectbox('Clip outliers',list(_clip_opts),index=0,key='validation_proxy_value_clip',help='Trim both axes by removing the top and bottom N% of data values.')]
                                    _px=pd.to_numeric(proxy_df[x_col],errors='coerce').dropna()
                                    _py=pd.to_numeric(proxy_df[proxy],errors='coerce').dropna() if proxy in proxy_df else pd.Series(dtype=float)
                                    proxy_x_range=[float(_px.quantile(_clip/100)),float(_px.quantile(1-_clip/100))] if _clip and not _px.empty else None
                                    proxy_y_range=[float(_py.quantile(_clip/100)),float(_py.quantile(1-_clip/100))] if _clip and not _py.empty else None
                                fig,_proxy_formulas=validation_proxy_figure(proxy_df,proxy,x_col,proxy_color,proxy_point_size,show_proxy_formulas,proxy_trend,proxy_trend_bin,proxy_trend_min_n,proxy_x_range,proxy_y_range)
                        else:
                            with st.expander('Proxy thickness controls',expanded=True):
                                default_methods=[m for m in ['H_Sundell2021_Paired_km','H_GAME_LuffiDucea2022_km','H_Sundell2021_SrY_km','H_Sundell2021_LaYbN_km','H_Hu2017_Collisional_SrY_km','H_Profeta2015_SrY_km','H_Mantle2008_CeY_sample_km'] if m in proxy_thickness_options]
                                methods=proxy_library_multiselect(proxy_thickness_options,default=default_methods or proxy_thickness_options[:4],key='validation_proxy_methods')
                                pt1,pt2,pt3,pt4=st.columns([0.45,1,1,1])
                                proxy_thickness_point_size=typed_slider(pt1,'Size',2,12,5,1,key='validation_proxy_thickness_point_size')
                                proxy_thickness_trend=pt2.selectbox('Moving summary',['Median','Mean','Both','Off'],index=0,key='validation_proxy_thickness_trend')
                                proxy_thickness_trend_bin=typed_slider(pt3,'Summary bin width [Km]',2.0,20.0,5.0,1.0,key='validation_proxy_thickness_trend_bin')
                                proxy_thickness_trend_min_n=typed_slider(pt4,'Minimum N',1,50,5,1,key='validation_proxy_thickness_trend_min_n')
                                _tclip_opts={'None':0,'0.5%':0.5,'1%':1.0,'2%':2.0,'5%':5.0}
                                _tclip=_tclip_opts[st.selectbox('Clip outliers',list(_tclip_opts),index=0,key='validation_thick_clip',help='Trim both axes by removing the top and bottom N% of data values.')]
                                _tx=pd.to_numeric(proxy_df[x_col],errors='coerce').dropna()
                                _ty_vals=[pd.to_numeric(proxy_df[m],errors='coerce').dropna() for m in methods if m in proxy_df]
                                _ty=pd.concat(_ty_vals) if _ty_vals else pd.Series(dtype=float)
                                thick_x_range=[float(_tx.quantile(_tclip/100)),float(_tx.quantile(1-_tclip/100))] if _tclip and not _tx.empty else None
                                thick_y_range=[float(_ty.quantile(_tclip/100)),float(_ty.quantile(1-_tclip/100))] if _tclip and not _ty.empty else None
                            fig,_proxy_formulas=validation_proxy_thickness_figure(proxy_df,x_col,methods,proxy_thickness_point_size,show_proxy_formulas,proxy_thickness_trend,proxy_thickness_trend_bin,proxy_thickness_trend_min_n,thick_x_range,thick_y_range)
                        if fig is not None:
                            st.plotly_chart(fig,width='stretch')
                            if _proxy_formulas:
                                for _fname, _fform in _proxy_formulas.items():
                                    st.caption(f'**{_fname}**: {_fform}')
                        else:
                            st.info('No complete proxy rows are available after filtering.')
                    else:
                        st.info('Proxy comparison needs a recognised proxy value or proxy-thickness method in the validation data.')

                    with st.expander('Local crustal-thickness estimate and grouping workflow',expanded=True):
                        st.caption('Estimate local crustal thickness for a target point, CRUST1.0 cell, sample, or arc segment from nearby geochemical samples in a combined spatial + temporal neighbourhood. Nearest-N, radius, and time window define the candidate population; they are not error bars.')
                        if {'Lat','Lon'}.issubset(test_bench):
                            domain_priority=[c for c in ['Arc_or_Segment','Geologic_Domain','Tectonic_Setting','Dataset','Rock_Type_Model','Geologic_Era','Geologic_Period','Geologic_Epoch'] if c in test_bench]
                            extra_domains=[c for c in test_bench.columns if c not in domain_priority and c not in ['Model','Algorithm'] and test_bench[c].dtype == object and 1 < test_bench[c].nunique(dropna=True) <= 30]
                            local_source_df=test_bench.copy()

                            st.markdown('**1. Target**')
                            tg1,tg2,tg3,tg4=st.columns(4)
                            target_mode=tg1.selectbox('Target type',['CRUST1.0 grid cell','Sample point','User-defined point','Arc segment'],index=0,key='local_target_mode')
                            user_lat=tg2.number_input('Target latitude',-90.0,90.0,0.0,0.1,key='local_user_lat',disabled=target_mode!='User-defined point')
                            user_lon=tg3.number_input('Target longitude',-180.0,180.0,0.0,0.1,key='local_user_lon',disabled=target_mode!='User-defined point')
                            user_age=tg4.number_input('Target age [Ma]',0.0,4500.0,0.0,1.0,key='local_user_age',disabled=target_mode!='User-defined point')
                            arc_group_options=domain_priority + extra_domains[:8]
                            arc_col=st.selectbox('Arc/segment field',arc_group_options,index=0,key='local_arc_col',disabled=target_mode!='Arc segment') if arc_group_options else None

                            st.markdown('**2. Time window**')
                            tw1,tw2,tw3=st.columns(3)
                            time_mode=tw1.selectbox('Time mode',['Modern benchmark: Age_Ma <= 5','Fixed time slice','Age bins','Ignore age'],index=0,key='local_time_mode')
                            local_time_window=typed_slider(tw2,'Fixed slice width [Ma]',1.0,200.0,10.0,1.0,key='local_time_window',disabled=time_mode!='Fixed time slice')
                            local_age_bin=typed_slider(tw3,'Age bin width [Ma]',1.0,500.0,10.0,1.0,key='local_age_bin_width',disabled=time_mode!='Age bins')

                            st.markdown('**3. Spatial neighbourhood**')
                            sp1,sp2,sp3,sp4=st.columns(4)
                            spatial_mode=sp1.selectbox('Spatial mode',['Radius with minimum-N fallback','Radius window','Nearest N'],index=0,key='local_spatial_mode')
                            local_initial_radius=typed_slider(sp2,'Initial radius [Km]',25,500,100,25,key='local_initial_radius')
                            local_max_radius=typed_slider(sp3,'Maximum radius [Km]',int(local_initial_radius),1000,max(250,int(local_initial_radius)),25,key='local_max_radius',disabled=spatial_mode!='Radius with minimum-N fallback')
                            local_nearest_n=typed_slider(sp4,'Nearest N',3,100,30,1,key='local_nearest_n',disabled=spatial_mode!='Nearest N')
                            sp5,sp6,sp7=st.columns(3)
                            local_radius_step=typed_slider(sp5,'Radius step [Km]',10,200,50,10,key='local_radius_step',disabled=spatial_mode!='Radius with minimum-N fallback')
                            local_min_n=typed_slider(sp6,'Minimum N',3,50,10,1,key='local_min_n')
                            local_preferred_n=typed_slider(sp7,'Preferred N',int(local_min_n),100,max(30,int(local_min_n)),1,key='local_preferred_n')

                            domain_mode=st.radio('Domain radius option',['Manual radius','Upload geological domain','Sample distribution long axis'],horizontal=True,key='local_domain_mode')
                            local_domain=None
                            radius_col=None
                            uploaded_records=[]
                            sample_group_col=None
                            if domain_mode == 'Upload geological domain':
                                domain_file=st.file_uploader('Upload geological domain polygon',type=['geojson','json','csv','xlsx','xls'],key='local_domain_upload',help='GeoJSON polygons, or a table with Domain, Lat, Lon polygon vertices.')
                                uploaded_records=read_domain_records(domain_file) if domain_file else []
                                if uploaded_records:
                                    local_source_df=attach_uploaded_geological_domains(local_source_df,uploaded_records)
                                    local_domain='Geological_Domain'
                                    radius_col='Geological_Domain_Long_Axis_km'
                                    table_action_card('Uploaded geological domains',tidy_numbers(pd.DataFrame([{k:v for k,v in r.items() if k != 'Polygon'} for r in uploaded_records])),'uploaded_geological_domains.csv','uploaded_geological_domains')
                                else:
                                    st.info('Upload a GeoJSON polygon file or a Domain/Lat/Lon vertex table to use polygon long-axis radii.')
                            elif domain_mode == 'Sample distribution long axis':
                                sample_group_options=['All samples'] + domain_priority + extra_domains[:8]
                                sample_group=st.selectbox('Build sample-distribution domain from',sample_group_options,index=0,key='local_sample_distribution_group')
                                sample_group_col=None if sample_group == 'All samples' else sample_group
                                local_source_df=attach_sample_distribution_domains(local_source_df,sample_group_col)
                                local_domain=None if sample_group_col is None else 'Sample_Distribution_Domain'
                                radius_col='Sample_Distribution_Long_Axis_km'
                                table_action_card('Sample-distribution domains',tidy_numbers(local_source_df[['Sample_Distribution_Domain','Sample_Distribution_Long_Axis_km']].drop_duplicates().sort_values('Sample_Distribution_Domain')),'sample_distribution_domains.csv','sample_distribution_domains')

                            st.markdown('**4. Filters**')
                            f1,f2,f3,f4=st.columns(4)
                            local_value_options=[c for c in ['Predicted_km'] + [p for p in PROXY_THICKNESS_LABELS if p in local_source_df] + ['Observed_km'] if c in local_source_df]
                            local_value=f1.selectbox('Geochemical H estimate',local_value_options,index=0,key='local_value',format_func=lambda c: {'Predicted_km':'model: crustal thickness [Km]','Observed_km':'known: crustal thickness [Km]'}.get(c,PROXY_THICKNESS_LABELS.get(c,c)))
                            rock_filter_options=sorted(local_source_df['Rock_Type_Model'].dropna().astype(str).unique()) if 'Rock_Type_Model' in local_source_df else []
                            selected_local_rocks=f2.multiselect('Rock type',rock_filter_options,default=rock_filter_options,key='local_rocks') if rock_filter_options else None
                            exclude_flagged=f3.checkbox('Exclude flagged samples',True,key='local_exclude_flagged')
                            complete_guo=f4.checkbox('Complete Guo rows only',False,key='local_complete_guo')
                            f5,f6=st.columns(2)
                            exclude_high_loi=f5.checkbox('Exclude high LOI',False,key='local_exclude_loi')
                            loi_max=typed_slider(f6,'Maximum LOI',0.0,20.0,5.0,0.5,key='local_loi_max',disabled=not exclude_high_loi)
                            local_candidates=local_source_df.copy()
                            if exclude_flagged and 'Reliability_Flags' in local_candidates:
                                local_candidates=local_candidates[local_candidates['Reliability_Flags'].fillna('').astype(str).str.strip().isin(['','OK','ok','Ok'])]
                            if exclude_high_loi and 'LOI' in local_candidates:
                                local_candidates=local_candidates[pd.to_numeric(local_candidates['LOI'],errors='coerce').le(loi_max) | local_candidates['LOI'].isna()]
                            if complete_guo:
                                ok_complete,_=complete(local_candidates,GUO_FEATURES)
                                local_candidates=local_candidates[ok_complete]

                            map_selected_available=False
                            if {'Lat','Lon'}.issubset(local_candidates):
                                with st.expander('Create grouping from map selection',expanded=False):
                                    st.caption('Use the Plotly lasso or box select tools on the map, then include Map-selected group in the grouping comparison. The selected points are treated as one temporary domain.')
                                    selectable=local_candidates.dropna(subset=['Lat','Lon']).reset_index(drop=True)
                                    if selectable.empty:
                                        st.info('No selectable Lat/Lon rows after filters.')
                                    else:
                                        sel_color_options=[c for c in [local_value,'Predicted_km'] + [p for p in PROXY_THICKNESS_LABELS if p in selectable] + ['Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset'] if c in selectable and (not pd.api.types.is_numeric_dtype(selectable[c]) or pd.to_numeric(selectable[c],errors='coerce').notna().any())]
                                        sel_color_options=list(dict.fromkeys(sel_color_options))
                                        sel_color=st.selectbox('Map selection colour by',sel_color_options,index=0,key='local_group_select_map_color',format_func=local_option_label)
                                        sel_hover=hover_cols(selectable,['Sample_ID','Age_Ma','Dataset','Arc_or_Segment','Geologic_Domain','Rock_Type_Model','Model',local_value,'Predicted_km','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb'] + [p for p in PROXY_THICKNESS_LABELS if p in selectable],sel_color,'Lat','Lon')
                                        sel_fig=px.scatter_geo(
                                            tidy_numbers(selectable),lat='Lat',lon='Lon',color=sel_color,
                                            hover_data=map_hover_data(selectable,sel_hover),
                                            projection='natural earth',template='plotly_white',
                                            labels={sel_color:local_option_label(sel_color)}
                                        )
                                        sel_fig.update_traces(marker=dict(size=7,line=dict(color='black',width=0.5)),selector=dict(type='scattergeo'))
                                        sel_fig.update_layout(height=420,dragmode='lasso',margin=dict(l=10,r=10,t=10,b=10))
                                        try:
                                            sel_event=st.plotly_chart(sel_fig,width='stretch',key='local_group_select_map',on_select='rerun',selection_mode=['lasso','box'])
                                            selected_idx=plotly_selected_indices(sel_event)
                                        except TypeError:
                                            st.plotly_chart(sel_fig,width='stretch')
                                            selected_idx=[]
                                            st.info('This Streamlit version does not expose map selections; use uploaded geological domains or auto grouping instead.')
                                        if selected_idx:
                                            selected_idx=[i for i in selected_idx if i < len(selectable)]
                                            selected_ids=set(selectable.iloc[selected_idx].index)
                                            selectable['Map_Selected_Group']='Outside selection'
                                            selectable.loc[list(selected_ids),'Map_Selected_Group']='Selected polygon'
                                            local_candidates=selectable
                                            map_selected_available=True
                                            st.success(f'{len(selected_idx)} samples selected for Map-selected group.')
                                        else:
                                            st.caption('No selected samples yet.')

                            st.markdown('**5. Grouping workflow**')
                            with st.expander('Grouping controls',expanded=True):
                                preset_group_options=['All samples'] + [c for c in ['Geologic_Domain','Arc_or_Segment','Tectonic_Setting','Dataset','Rock_Type_Model','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Age_Ma'] if c in local_candidates]
                                preset_group_options += [c for c in extra_domains[:8] if c in local_candidates and c not in preset_group_options]
                                preset_group_options=list(dict.fromkeys(preset_group_options))
                                default_preset_group='Geologic_Domain' if 'Geologic_Domain' in preset_group_options else 'Arc_or_Segment' if 'Arc_or_Segment' in preset_group_options else 'Age_Ma' if 'Age_Ma' in preset_group_options else 'All samples'
                                pg1,pg2,pg3,pg4=st.columns([1,0.75,0.8,0.8])
                                preset_group_col=pg1.selectbox(
                                    'Preset group by',
                                    preset_group_options,
                                    index=preset_group_options.index(default_preset_group) if default_preset_group in preset_group_options else 0,
                                    key='local_preset_group_col',
                                    format_func=lambda c: 'All samples' if c == 'All samples' else 'Belt / arc / segment' if c == 'Arc_or_Segment' else 'Age bins' if c == 'Age_Ma' else local_option_label(c),
                                )
                                preset_split_distance=pg2.checkbox('Sub-split by distance',True,key='local_preset_split_distance')
                                preset_segment_km=typed_slider(pg3,'Max segment [Km]',10,1000,100,10,key='local_preset_segment_km',disabled=not preset_split_distance)
                                preset_age_bin_width=typed_slider(pg4,'Age bin [Ma]',1.0,500.0,50.0,1.0,key='local_preset_age_bin_width',disabled=preset_group_col!='Age_Ma')
                                compare_options=['Preset grouping','Manual radius','Sample distribution long axis','Auto KMeans','Auto DBSCAN','Auto Agglomerative']
                                if uploaded_records:
                                    compare_options.insert(1,'Uploaded geological domain')
                                if map_selected_available:
                                    compare_options.insert(1,'Map-selected group')
                                default_compare=['Preset grouping']
                                if domain_mode in compare_options and domain_mode not in default_compare:
                                    default_compare.append(domain_mode)
                                if 'Auto KMeans' in compare_options and 'Auto KMeans' not in default_compare:
                                    default_compare.append('Auto KMeans')
                                grouping_methods=st.multiselect('Grouping methods to compare',compare_options,default=default_compare,key='local_grouping_methods')
                                ac1,ac2,ac3,ac4=st.columns(4)
                                proxy_h_features=[c for c in PROXY_THICKNESS_LABELS if c in local_candidates]
                                auto_features_available=[c for c in ['Age_Ma',local_value,'Predicted_km'] + proxy_h_features + ['Sr_Y','La_Yb_N','Ce_Y','SiO2','MgO','La','Yb','Sr','Y'] if c in local_candidates]
                                auto_features_available=list(dict.fromkeys([c for c in auto_features_available if not str(c).startswith('CRUST1') and c != 'Observed_km' and c != 'Residual_km']))
                                default_auto=[c for c in ['Age_Ma',local_value,'Predicted_km','H_Sundell2021_Paired_km','H_Sundell2021_SrY_km','H_Sundell2021_LaYbN_km','Sr_Y','La_Yb_N'] if c in auto_features_available]
                                auto_features=ac1.multiselect('Auto-domain features',auto_features_available,default=list(dict.fromkeys(default_auto)),key='local_auto_features',format_func=lambda c: {'Predicted_km':'model: crustal thickness [Km]','Age_Ma':'Age [Ma]','Sr_Y':'Sr/Y','La_Yb_N':'La/Yb(N)','Ce_Y':'Ce/Y'}.get(c,PROXY_THICKNESS_LABELS.get(c,c)))
                                auto_k_mode=ac2.selectbox('Auto K mode',['Max range','Manual K'],index=0,key='local_auto_k_mode')
                                auto_k=typed_slider(ac2,'Starting K',2,12,4,1,key='local_auto_k')
                                auto_eps=typed_slider(ac3,'DBSCAN eps',0.2,3.0,0.9,0.1,key='local_auto_eps')
                                auto_min=typed_slider(ac4,'Auto min samples',3,30,8,1,key='local_auto_min')
                                rk1,rk2,rk3=st.columns(3)
                                max_group_range=typed_slider(rk1,'Max group long-axis [Km]',50,2000,500,50,key='local_auto_max_group_range',disabled=auto_k_mode!='Max range')
                                max_h_range=typed_slider(rk2,'Max group H range [Km]',0.0,60.0,0.0,1.0,key='local_auto_max_h_range',disabled=auto_k_mode!='Max range',help='0 disables this constraint.')
                                max_auto_k=typed_slider(rk3,'Maximum K',2,30,12,1,key='local_auto_max_k',disabled=auto_k_mode!='Max range')
                                aw1,aw2,aw3,aw4=st.columns(4)
                                spatial_weight=typed_slider(aw1,'Spatial weight',0.0,5.0,1.0,0.25,key='local_auto_spatial_weight')
                                age_weight=typed_slider(aw2,'Age weight',0.0,5.0,1.0,0.25,key='local_auto_age_weight')
                                thickness_weight=typed_slider(aw3,'Thickness weight',0.0,5.0,1.0,0.25,key='local_auto_thickness_weight')
                                chemistry_weight=typed_slider(aw4,'Chemistry weight',0.0,5.0,0.5,0.25,key='local_auto_chemistry_weight')

                            local_boot=typed_slider(st,'Bootstrap repeats',100,2000,500,100,key='local_boot')
                            local_frames=[]
                            domain_summaries=[]
                            grouping_map_frames=[]
                            for gm in grouping_methods:
                                gm_source=local_candidates.copy()
                                gm_domain=None
                                gm_radius_col=None
                                if gm == 'Preset grouping':
                                    base_col=None if preset_group_col == 'All samples' else preset_group_col
                                    gm_source=attach_preset_group_domains(
                                        gm_source,
                                        base_col,
                                        max_segment_km=preset_segment_km,
                                        split_by_distance=preset_split_distance,
                                        age_bin_width=preset_age_bin_width,
                                    )
                                    gm_domain='Preset_Group_ID'
                                    gm_radius_col='Preset_Group_Segment_Long_Axis_km'
                                    domain_summaries.append(preset_group_summary(gm_source))
                                elif gm == 'Uploaded geological domain' and uploaded_records:
                                    gm_source=attach_uploaded_geological_domains(gm_source,uploaded_records)
                                    gm_domain='Geological_Domain'
                                    gm_radius_col='Geological_Domain_Long_Axis_km'
                                elif gm == 'Map-selected group' and 'Map_Selected_Group' in gm_source:
                                    gm_domain='Map_Selected_Group'
                                    gm_radius_col=None
                                elif gm == 'Sample distribution long axis':
                                    gm_source=attach_sample_distribution_domains(gm_source,sample_group_col)
                                    gm_domain=None if sample_group_col is None else 'Sample_Distribution_Domain'
                                    gm_radius_col='Sample_Distribution_Long_Axis_km'
                                elif gm.startswith('Auto '):
                                    auto_method=gm.replace('Auto ','')
                                    gm_source, gm_domain, auto_summary = attach_auto_domains(
                                        gm_source, tuple(auto_features), method=auto_method, n_clusters=auto_k, eps=auto_eps,
                                        min_samples=auto_min, spatial_weight=spatial_weight, age_weight=age_weight,
                                        thickness_weight=thickness_weight, chemistry_weight=chemistry_weight,
                                        label=gm, auto_k_by_range=auto_k_mode=='Max range',
                                        max_group_range_km=max_group_range, max_h_range_km=max_h_range,
                                        h_col=local_value if local_value in gm_source else 'Predicted_km',
                                        max_clusters=max_auto_k
                                    )
                                    domain_summaries.append(auto_summary)
                                else:
                                    gm_domain=None
                                    gm_radius_col=None
                                if gm_domain and gm_domain in gm_source and {'Lat','Lon'}.issubset(gm_source):
                                    gm_map=gm_source.dropna(subset=['Lat','Lon']).copy()
                                    gm_map['Grouping_Method']=gm
                                    gm_map['Group_ID']=gm_map[gm_domain].astype(str)
                                    grouping_map_frames.append(gm_map)
                                gm_targets=build_local_targets(target_mode,gm_source,user_lat,user_lon,user_age,arc_col)
                                gm_targets=attach_target_domains_from_samples(gm_targets,gm_source,gm_domain,gm_radius_col)
                                if target_mode == 'CRUST1.0 grid cell' and gm_targets.empty:
                                    continue
                                gm_df=local_crustal_thickness_estimates(
                                    gm_targets,gm_source,local_value,time_mode,user_age,local_time_window,local_age_bin,
                                    spatial_mode,local_initial_radius,local_max_radius,local_radius_step,local_nearest_n,
                                    local_min_n,local_preferred_n,tuple(selected_local_rocks) if selected_local_rocks else None,gm_domain,gm_radius_col,local_boot,seed
                                )
                                if not gm_df.empty:
                                    gm_df.insert(0,'Grouping_Method',gm)
                                    local_frames.append(gm_df)
                            local_df=pd.concat(local_frames,ignore_index=True) if local_frames else pd.DataFrame()
                            if domain_summaries:
                                auto_summary_df=pd.concat([d for d in domain_summaries if not d.empty],ignore_index=True) if any(not d.empty for d in domain_summaries) else pd.DataFrame()
                                if not auto_summary_df.empty:
                                    table_action_card('Grouping summary',auto_summary_df,'grouping_summary.csv','grouping_summary')
                                    if 'K_Selection' in auto_summary_df and auto_summary_df['K_Selection'].astype(str).str.contains('not fully met',case=False,na=False).any():
                                        st.warning('Auto grouping reached the maximum K before every group met the selected range target. Increase Maximum K or relax the max range.')
                            if grouping_map_frames:
                                group_map=pd.concat(grouping_map_frames,ignore_index=True)
                                st.session_state['_val_group_map']=group_map.copy()
                                st.session_state['_val_local_df']=local_df.copy() if not local_df.empty else pd.DataFrame()
                                st.session_state['_val_local_value']=local_value
                                st.markdown('**6. Grouping diagnostics**')
                                _render_grouping_display('val')
                            elif grouping_methods:
                                st.info('Select at least one non-manual grouping method, such as Auto KMeans or Sample distribution long axis, to show the grouping map and proxy graph.')
                            if target_mode == 'CRUST1.0 grid cell' and local_df.empty:
                                st.info('CRUST1.0 targets need the default CRUST_1_0_excel.csv grid plus validation Lat/Lon points.')
                            if local_df.empty:
                                st.info('No local crustal-thickness estimates could be calculated. Check target coordinates, filters, Age_Ma, and selected model value availability.')
                            else:
                                st.markdown('**7. Estimate statistics**')
                                if {'Grouping_Method','Local_Domain_Value','Local_Median_H_km'}.issubset(local_df):
                                    _grp_cols=[c for c in ['Grouping_Method','Local_Domain_Value','Model'] if c in local_df]
                                    _grp_aggs={'Local_Median_H_km':['count','median']}
                                    for _ac in ['Local_Mean_H_km','Bootstrap_95CI_Low_Median','Bootstrap_95CI_High_Median','Local_N','Local_IQR_H_km','Local_Radius_km','Residual_Local_vs_Known_Thickness_km','Residual_Local_vs_CRUST1_Total_km','Residual_Local_vs_CRUST1_Crystalline_km']:
                                        if _ac in local_df:
                                            _grp_aggs[_ac]='median'
                                    if 'Good_Local_Estimate' in local_df:
                                        _grp_aggs['Good_Local_Estimate']='mean'
                                    _domain_summary=local_df.groupby(_grp_cols,dropna=False).agg(_grp_aggs).reset_index()
                                    _domain_summary.columns=['_'.join([str(x) for x in c if x]) for c in _domain_summary.columns]
                                    _rename={'Local_Median_H_km_count':'N_targets','Local_Median_H_km_median':'Median_H_km'}
                                    _domain_summary=_domain_summary.rename(columns=_rename)
                                    table_action_card('Group-level crustal-thickness summary',tidy_numbers(_domain_summary),'local_group_summary.csv','local_group_summary')
                                    with st.expander('Individual target rows',expanded=False):
                                        lead_cols=[c for c in ['Grouping_Method','Target_Type','Target_ID','Model','Sample_ID','Dataset','Arc_or_Segment','Geologic_Domain','Local_Domain_Column','Local_Domain_Value','Age_Ma','Lat','Lon','Local_Median_H_km','Local_Mean_H_km','Bootstrap_95CI_Low_Median','Bootstrap_95CI_High_Median','Local_N','Local_Radius_km','Local_Time_Window_Ma','Local_Radius_Source','Local_Domain_Long_Axis_km','CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Sediment_km','Residual_Local_vs_CRUST1_Total_km','Residual_Local_vs_CRUST1_Crystalline_km','Residual_Local_vs_Known_Thickness_km','Good_Local_Estimate','Low_N','Expanded_Radius','High_IQR','High_Bootstrap_Uncertainty','High_Sediment_Cover','No_Estimate'] if c in local_df]
                                        rest_cols=[c for c in local_df.columns if c not in lead_cols]
                                        table_action_card('Local crustal-thickness estimates',local_df[lead_cols+rest_cols],'local_crustal_thickness_estimates.csv','local_estimate_statistics')
                                else:
                                    lead_cols=[c for c in ['Grouping_Method','Target_Type','Target_ID','Model','Sample_ID','Dataset','Arc_or_Segment','Geologic_Domain','Local_Domain_Column','Local_Domain_Value','Age_Ma','Lat','Lon','Local_Median_H_km','Local_Mean_H_km','Bootstrap_95CI_Low_Median','Bootstrap_95CI_High_Median','Local_N','Local_Radius_km','Local_Time_Window_Ma','Local_Radius_Source','Local_Domain_Long_Axis_km','CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Sediment_km','Residual_Local_vs_CRUST1_Total_km','Residual_Local_vs_CRUST1_Crystalline_km','Residual_Local_vs_Known_Thickness_km','Good_Local_Estimate','Low_N','Expanded_Radius','High_IQR','High_Bootstrap_Uncertainty','High_Sediment_Cover','No_Estimate'] if c in local_df]
                                    rest_cols=[c for c in local_df.columns if c not in lead_cols]
                                    table_action_card('Local crustal-thickness estimates',local_df[lead_cols+rest_cols],'local_crustal_thickness_estimates.csv','local_estimate_statistics')
                                plot_local=local_df.dropna(subset=['Local_Median_H_km'])
                                if not plot_local.empty:
                                    plot_local=plot_local.copy().reset_index(drop=True)
                                    plot_local['Target_Index'] = np.arange(1,len(plot_local)+1)
                                    fig=go.Figure()
                                    y_options=[c for c in ['Local_Median_H_km','Local_Mean_H_km','Local_Q25_H_km','Local_Q75_H_km','Local_Min_H_km','Local_Max_H_km'] if c in plot_local and pd.to_numeric(plot_local[c],errors='coerce').notna().any()]
                                    y_label_map={'Local_Median_H_km':'Local_Median_H_km','Local_Mean_H_km':'Local_Mean_H_km','Local_Q25_H_km':'Local_Q25_H_km','Local_Q75_H_km':'Local_Q75_H_km','Local_Min_H_km':'Local_Min_H_km','Local_Max_H_km':'Local_Max_H_km'}
                                    y_col=st.selectbox('Local estimate Y axis',y_options,index=0,key='local_estimate_y',format_func=lambda c: y_label_map.get(c,c))
                                    x_candidates=[]
                                    if target_mode == 'CRUST1.0 grid cell':
                                        x_candidates += [c for c in ['CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km'] if c in plot_local and pd.to_numeric(plot_local[c],errors='coerce').notna().any()]
                                    local_median_proxy_cols=[c for c in plot_local.columns if str(c).startswith('Local_Median_') and pd.to_numeric(plot_local[c],errors='coerce').notna().any()]
                                    x_candidates += [c for c in [local_value,'Observed_km','Predicted_km'] + [p for p in PROXY_THICKNESS_LABELS if p in plot_local] + local_median_proxy_cols + ['Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2','Target_Index'] if c in plot_local and pd.to_numeric(plot_local[c],errors='coerce').notna().any()]
                                    x_candidates=list(dict.fromkeys(x_candidates)) or ['Target_Index']
                                    x_label_map={'CRUST1_Total_Crust_km':'CRUST1.0: total crustal thickness [Km]','CRUST1_Crystalline_Crust_km':'CRUST1.0: crystalline crustal thickness [Km]','Predicted_km':'model: crustal thickness [Km]','Observed_km':'known: crustal thickness [Km]','Residual_km':'model - known [Km]','Age_Ma':'Age [Ma]','Target_Index':'target index'}
                                    x_col=st.selectbox('Local estimate X axis',x_candidates,index=0,key='local_estimate_x',format_func=lambda c: x_label_map.get(c,local_option_label(c)))
                                    color_options=[c for c in ['Grouping_Method','Local_Domain_Value',local_value,'Predicted_km'] + [p for p in PROXY_THICKNESS_LABELS if p in plot_local] + local_median_proxy_cols + ['Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2','Local_N','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset'] if c in plot_local and (c in ['Grouping_Method','Local_Domain_Value'] or not pd.api.types.is_numeric_dtype(plot_local[c]) or pd.to_numeric(plot_local[c],errors='coerce').notna().any())]
                                    color_options=list(dict.fromkeys(color_options))
                                    graph_color=st.selectbox('Colour points by',color_options,index=0,key='local_estimate_color',format_func=local_option_label)
                                    show_local_samples=st.checkbox('Show individual samples',False,key='local_estimate_show_samples')
                                    show_local_legend=st.checkbox('Show legend',False,key='local_estimate_show_legend')
                                    plot_local=plot_local.dropna(subset=[x_col,y_col])
                                    if plot_local.empty:
                                        st.info('Local medians were calculated, but the selected X axis has no numeric values.')
                                    else:
                                        plot_resolution_options=['Target/local medians']
                                        if {'Grouping_Method','Local_Domain_Value'}.issubset(plot_local):
                                            plot_resolution_options.insert(0,'Group medians')
                                        plot_resolution=st.selectbox('Plot resolution',plot_resolution_options,index=0,key='local_estimate_plot_resolution')
                                        if plot_resolution == 'Group medians':
                                            group_cols=[c for c in ['Grouping_Method','Local_Domain_Value','Model'] if c in plot_local]
                                            numeric_cols=[c for c in plot_local.columns if pd.to_numeric(plot_local[c],errors='coerce').notna().any()]
                                            rows=[]
                                            for group_key, g in plot_local.groupby(group_cols,dropna=False):
                                                if not isinstance(group_key, tuple):
                                                    group_key=(group_key,)
                                                row={c:v for c,v in zip(group_cols,group_key)}
                                                for c in numeric_cols:
                                                    row[c]=pd.to_numeric(g[c],errors='coerce').median()
                                                row['Group_Target_Count']=len(g)
                                                row['Good_Local_Estimate']=bool(g['Good_Local_Estimate'].astype(bool).mean() >= 0.5) if 'Good_Local_Estimate' in g else True
                                                rows.append(row)
                                            plot_local=pd.DataFrame(rows).dropna(subset=[x_col,y_col])
                                            if plot_local.empty:
                                                st.info('No group medians are available for the selected X/Y fields.')
                                                for c in ['Bootstrap_95CI_High_Median','Bootstrap_95CI_Low_Median','Local_Median_H_km','Local_Q75_H_km','Local_Q25_H_km','Local_SD_H_km','Local_Max_H_km','Local_Min_H_km','Local_N','Local_Radius_km','Good_Local_Estimate']:
                                                    if c not in plot_local:
                                                        plot_local[c]=pd.Series(dtype=float)
                                        x=plot_local[x_col]
                                        y=plot_local[y_col]
                                        err_plus=(plot_local['Bootstrap_95CI_High_Median']-plot_local['Local_Median_H_km']).clip(lower=0)
                                        err_minus=(plot_local['Local_Median_H_km']-plot_local['Bootstrap_95CI_Low_Median']).clip(lower=0)
                                        spread=st.selectbox('Optional spread envelope',['Bootstrap 95% CI on median','IQR','+/- 1 SD','+/- 2 SD','Min-max','None'],index=0,key='local_spread')
                                        error_y=None
                                        if spread == 'Bootstrap 95% CI on median':
                                            error_y=dict(type='data',array=err_plus,arrayminus=err_minus,thickness=0.8,width=2,color='rgba(40,40,40,0.45)')
                                        elif spread == 'IQR':
                                            error_y=dict(type='data',array=(plot_local['Local_Q75_H_km']-plot_local['Local_Median_H_km']).clip(lower=0),arrayminus=(plot_local['Local_Median_H_km']-plot_local['Local_Q25_H_km']).clip(lower=0),thickness=0.8,width=2,color='rgba(40,40,40,0.45)')
                                        elif spread == '+/- 1 SD':
                                            error_y=dict(type='data',array=plot_local['Local_SD_H_km'],arrayminus=plot_local['Local_SD_H_km'],thickness=0.8,width=2,color='rgba(40,40,40,0.45)')
                                        elif spread == '+/- 2 SD':
                                            error_y=dict(type='data',array=2*plot_local['Local_SD_H_km'],arrayminus=2*plot_local['Local_SD_H_km'],thickness=0.8,width=2,color='rgba(40,40,40,0.45)')
                                        elif spread == 'Min-max':
                                            error_y=dict(type='data',array=(plot_local['Local_Max_H_km']-plot_local['Local_Median_H_km']).clip(lower=0),arrayminus=(plot_local['Local_Median_H_km']-plot_local['Local_Min_H_km']).clip(lower=0),thickness=0.8,width=2,color='rgba(40,40,40,0.45)')
                                        numeric_graph_color = graph_color in plot_local and graph_color != 'Grouping_Method' and pd.to_numeric(plot_local[graph_color],errors='coerce').notna().any()
                                        if graph_color in plot_local and not numeric_graph_color and graph_color != 'Grouping_Method' and 'Grouping_Method' in plot_local:
                                            group_items = list(plot_local.groupby(['Grouping_Method', graph_color], dropna=False))
                                        elif 'Grouping_Method' in plot_local:
                                            group_items = list(plot_local.groupby('Grouping_Method',dropna=False))
                                        else:
                                            group_items = [('Local median H',plot_local)]
                                        for group_i, (method_name, method_df) in enumerate(group_items):
                                            trace_name = ' / '.join([str(v) for v in method_name]) if isinstance(method_name, tuple) else str(method_name)
                                            method_error_y=None
                                            if error_y is not None:
                                                idx=method_df.index
                                                method_error_y=dict(
                                                    type='data',
                                                    array=pd.Series(error_y['array'],index=plot_local.index).loc[idx],
                                                    arrayminus=pd.Series(error_y['arrayminus'],index=plot_local.index).loc[idx],
                                                    thickness=0.8,width=2,color='rgba(40,40,40,0.45)'
                                                )
                                            marker=dict(size=8,line=dict(color='black',width=0.7))
                                            if numeric_graph_color:
                                                marker=dict(size=8,line=dict(color='black',width=0.7),color=pd.to_numeric(method_df[graph_color],errors='coerce'),colorscale='Viridis',showscale=show_local_legend and group_i == 0,colorbar=dict(title=local_option_label(graph_color)))
                                            fig.add_trace(go.Scatter(
                                                x=method_df[x_col],y=method_df[y_col],mode='markers',
                                                name=trace_name,
                                                marker=marker,
                                                showlegend=show_local_legend,
                                                error_y=method_error_y,
                                                customdata=np.c_[method_df['Local_N'],method_df['Local_Radius_km'],method_df['Good_Local_Estimate'],method_df.get(graph_color,pd.Series(np.nan,index=method_df.index))],
                                                hovertemplate=local_option_label(x_col)+'=%{x:.1f}<br>'+y_label_map.get(y_col,y_col)+'=%{y:.1f} Km<br>Local N=%{customdata[0]:.0f}<br>radius=%{customdata[1]:.1f} Km<br>Good=%{customdata[2]}<br>'+local_option_label(graph_color)+'=%{customdata[3]}<extra>'+trace_name+'</extra>'
                                            ))
                                        if show_local_samples:
                                            sample_y = local_value if local_value in local_candidates else 'Predicted_km'
                                            sample_x = x_col
                                            if sample_x.startswith('Local_Median_'):
                                                sample_x = sample_x.replace('Local_Median_','')
                                            if sample_x in local_candidates and sample_y in local_candidates:
                                                sample_overlay=local_candidates.dropna(subset=[sample_x,sample_y]).copy()
                                                if not sample_overlay.empty:
                                                    sample_color = graph_color
                                                    if sample_color.startswith('Local_Median_'):
                                                        sample_color = sample_color.replace('Local_Median_','')
                                                    if sample_color not in sample_overlay:
                                                        sample_color = None
                                                    sample_fig=px.scatter(
                                                        tidy_numbers(sample_overlay),x=sample_x,y=sample_y,color=sample_color,
                                                        hover_data=map_hover_data(sample_overlay,hover_cols(sample_overlay,['Sample_ID','Age_Ma','Dataset','Arc_or_Segment','Geologic_Domain','Rock_Type_Model','Model',sample_y,'Predicted_km','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb'] + [p for p in PROXY_THICKNESS_LABELS if p in sample_overlay],sample_x,sample_y,sample_color)),
                                                        template='plotly_white'
                                                    )
                                                    for tr in sample_fig.data:
                                                        tr.name=f'individual samples: {tr.name}' if sample_color else 'individual samples'
                                                        tr.showlegend=show_local_legend
                                                        tr.marker.size=4
                                                        tr.marker.line=dict(width=0)
                                                        tr.opacity=0.35
                                                        fig.add_trace(tr)
                                            else:
                                                st.caption('Individual samples need the selected X axis and geochemical H estimate to exist on the sample table.')
                                        if x_col in ['CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','Predicted_km','Observed_km']:
                                            fig.add_trace(go.Scatter(x=[0,90],y=[0,90],mode='lines',name='1:1',line=dict(color='black',dash='dash'),showlegend=show_local_legend))
                                        fig.update_layout(template='plotly_white',height=460,xaxis_title=x_label_map.get(x_col,local_option_label(x_col)),yaxis_title=y_label_map.get(y_col,y_col),margin=dict(l=20,r=20,t=20,b=20),showlegend=show_local_legend)
                                        st.plotly_chart(fig,width='stretch')

                                with st.expander('Trend plots',expanded=False):
                                    st.caption('One-dimensional moving views by age, longitude, latitude, or transect position. These are exploratory trend plots, not the local CRUST1.0 validation estimate.')
                                    axis_group_col=local_domain if local_domain and local_domain in plot_local else None
                                    trend_df=attach_long_axis_position(plot_local,axis_group_col)
                                    trend_source=attach_long_axis_position(local_candidates,axis_group_col)
                                    trend_plot_df=plot_df(tidy_numbers(trend_df))
                                    trend_source_plot=plot_df(tidy_numbers(trend_source))
                                    tr1,tr2,tr3,tr4=st.columns([1,1,1,0.8])
                                    trend_numeric_candidates=['Age_Ma','Long_Axis_Position_km','Lat','Lon','Local_Radius_km','Local_N','Local_Median_H_km','Local_Mean_H_km','Local_IQR_H_km','Local_SD_H_km','Predicted_km','Observed_km','Residual_km'] + [p for p in PROXY_THICKNESS_LABELS if p in trend_plot_df] + ['Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2']
                                    trend_x_options=[c for c in trend_numeric_candidates if has_numeric_column(trend_plot_df,c)]
                                    trend_y_options=[c for c in ['Local_Median_H_km','Local_Mean_H_km','Predicted_km','Observed_km'] + [p for p in PROXY_THICKNESS_LABELS if p in trend_plot_df] + ['Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','Local_N','Local_IQR_H_km','Local_SD_H_km'] if has_numeric_column(trend_plot_df,c)]
                                    trend_color_options=[c for c in ['Grouping_Method','Local_Domain_Value','Model','Local_N','Good_Local_Estimate',local_value,'Predicted_km'] + [p for p in PROXY_THICKNESS_LABELS if p in trend_plot_df] + ['Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset','Low_N','High_IQR','High_Bootstrap_Uncertainty'] if c in trend_plot_df and (c in ['Grouping_Method','Local_Domain_Value','Model','Good_Local_Estimate','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset','Low_N','High_IQR','High_Bootstrap_Uncertainty'] or has_numeric_column(trend_plot_df,c))]
                                    if not trend_x_options or not trend_y_options:
                                        st.info('Trend plots need at least one numeric X and Y field.')
                                    else:
                                        trend_x=tr1.selectbox('Trend X',trend_x_options,index=0,key='local_trend_x',format_func=local_option_label)
                                        trend_y=tr2.selectbox('Trend Y',trend_y_options,index=0,key='local_trend_y',format_func=local_option_label)
                                        trend_color=tr3.selectbox('Trend colour',trend_color_options,index=0,key='local_trend_color',format_func=local_option_label) if trend_color_options else ''
                                        show_samples=tr4.checkbox('Show samples',True,key='local_trend_show_samples')
                                        trend_work=trend_plot_df.copy()
                                        trend_work[trend_x]=numeric_series(trend_work,trend_x)
                                        trend_work[trend_y]=numeric_series(trend_work,trend_y)
                                        trend_work=trend_work.dropna(subset=[trend_x,trend_y])
                                        if trend_work.empty:
                                            st.info('No trend rows remain for the selected X/Y fields.')
                                        else:
                                            trend_fig=go.Figure()
                                            if trend_color in trend_work:
                                                color_num=pd.to_numeric(col_series(trend_work,trend_color),errors='coerce')
                                                if color_num.notna().any() and trend_color not in ['Grouping_Method','Local_Domain_Value','Model','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset']:
                                                    trend_fig.add_trace(go.Scatter(
                                                        x=trend_work[trend_x],y=trend_work[trend_y],mode='markers',name=local_option_label(trend_color),
                                                        marker=dict(size=8,color=color_num,colorscale='Viridis',colorbar=dict(title=local_option_label(trend_color)),line=dict(color='black',width=0.5)),
                                                        customdata=np.c_[color_num],
                                                        hovertemplate=local_option_label(trend_x)+'=%{x:.1f}<br>'+local_option_label(trend_y)+'=%{y:.1f}<br>'+local_option_label(trend_color)+'=%{customdata[0]:.1f}<extra></extra>'
                                                    ))
                                                else:
                                                    palette=px.colors.qualitative.Set2
                                                    for i,(name,g) in enumerate(trend_work.groupby(col_series(trend_work,trend_color).astype(str),dropna=False)):
                                                        trend_fig.add_trace(go.Scatter(
                                                            x=g[trend_x],y=g[trend_y],mode='markers',name=str(name),
                                                            marker=dict(size=8,color=palette[i % len(palette)],line=dict(color='black',width=0.5)),
                                                            hovertemplate=local_option_label(trend_x)+'=%{x:.1f}<br>'+local_option_label(trend_y)+'=%{y:.1f}<extra>'+str(name)+'</extra>'
                                                        ))
                                            else:
                                                trend_fig.add_trace(go.Scatter(
                                                    x=trend_work[trend_x],y=trend_work[trend_y],mode='markers',name='local estimates',
                                                    marker=dict(size=8,color='rgba(80,120,220,0.85)',line=dict(color='black',width=0.5)),
                                                    hovertemplate=local_option_label(trend_x)+'=%{x:.1f}<br>'+local_option_label(trend_y)+'=%{y:.1f}<extra></extra>'
                                                ))
                                            if show_samples and trend_x in trend_source_plot and local_value in trend_source_plot:
                                                raw_overlay=trend_source_plot.copy()
                                                raw_overlay[trend_x]=numeric_series(raw_overlay,trend_x)
                                                raw_overlay[local_value]=numeric_series(raw_overlay,local_value)
                                                raw_overlay=raw_overlay.dropna(subset=[trend_x,local_value])
                                                trend_fig.add_trace(go.Scatter(x=raw_overlay[trend_x],y=raw_overlay[local_value],mode='markers',name='individual samples',marker=dict(size=4,color='rgba(120,120,120,0.35)',line=dict(width=0)),hovertemplate=local_option_label(trend_x)+'=%{x:.1f}<br>'+local_option_label(local_value)+'=%{y:.1f}<extra>individual sample</extra>'))
                                            if trend_x == 'Age_Ma':
                                                if 'CRUST1_Total_Crust_km' in trend_plot_df:
                                                    ctot=trend_plot_df.dropna(subset=['CRUST1_Total_Crust_km'])
                                                    if not ctot.empty:
                                                        trend_fig.add_trace(go.Scatter(x=np.zeros(len(ctot)),y=ctot['CRUST1_Total_Crust_km'],mode='markers',name='CRUST1.0 total at 0 Ma',marker=dict(symbol='diamond',size=8,color='black'),hovertemplate='Age=0 Ma<br>CRUST1.0 total=%{y:.1f} Km<extra></extra>'))
                                                if 'CRUST1_Crystalline_Crust_km' in trend_plot_df:
                                                    ccry=trend_plot_df.dropna(subset=['CRUST1_Crystalline_Crust_km'])
                                                    if not ccry.empty:
                                                        trend_fig.add_trace(go.Scatter(x=np.zeros(len(ccry)),y=ccry['CRUST1_Crystalline_Crust_km'],mode='markers',name='CRUST1.0 crystalline at 0 Ma',marker=dict(symbol='diamond-open',size=8,color='black'),hovertemplate='Age=0 Ma<br>CRUST1.0 crystalline=%{y:.1f} Km<extra></extra>'))
                                                if 'Observed_km' in trend_plot_df:
                                                    known=trend_plot_df.dropna(subset=['Age_Ma','Observed_km'])
                                                    if not known.empty:
                                                        trend_fig.add_trace(go.Scatter(x=known['Age_Ma'],y=known['Observed_km'],mode='markers',name='known/reference thickness',marker=dict(symbol='x',size=8,color='firebrick'),hovertemplate='Age=%{x:.1f} Ma<br>known=%{y:.1f} Km<extra></extra>'))
                                            trend_fig.update_layout(height=430,margin=dict(l=20,r=20,t=20,b=20),template='plotly_white',xaxis_title=local_option_label(trend_x),yaxis_title=local_option_label(trend_y))
                                            st.plotly_chart(trend_fig,width='stretch')
                                st.download_button('Download local crustal-thickness estimates',local_df.to_csv(index=False).encode('utf-8'),'local_crustal_thickness_estimates.csv','text/csv')
                        else:
                            st.info('Local crustal-thickness estimates need Lat and Lon columns in the validation data.')

                    st.subheader('Blind validation map')
                    if {'Lat','Lon'}.issubset(test_bench):
                        map_options=[c for c in ['Predicted_km','Observed_km','Residual_km','CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Sediment_km','Age_Ma','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset','Rock_Type_Model'] if c in test_bench]
                        if map_options:
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
                        else:
                            st.info('No validation columns are available for map colouring.')
                    else:
                        st.info('Validation map needs Lat and Lon columns.')
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
    _dp_pred_ready=not st.session_state.get('dp_prediction_df',pd.DataFrame()).empty
    _pred_src_opts=['Upload file']+(['From Data Prep tab'] if _dp_pred_ready else [])
    _pred_src=st.radio('Prediction data source',_pred_src_opts,horizontal=True,key='pred_data_source')
    if _pred_src=='From Data Prep tab':
        uk_raw=st.session_state['dp_prediction_df'].copy()
        st.success(f"Using Data Prep prediction data: {len(uk_raw):,} rows.")
        uk_up=None
    else:
        uk_up=st.file_uploader('Upload unknown geochemistry dataset',type=['csv','xlsx','xls'],key='uk_upload')
        uk_raw=read_table(uk_up,guo_no_header=True,expected=TRAINING_COLUMNS) if uk_up else pd.DataFrame()
    if uk_raw.empty and _pred_src=='Upload file' and not uk_up:
        st.info('Upload a geochemistry dataset. A known crustal thickness column is not required.')
    elif not uk_raw.empty:
        if not models:
            st.info('Train at least one model in the Model tab to run predictions against this data.')
        else:
            uk_raw=training_subset_controls('Unknown dataset',uk_raw,expanded=True,target=None,noun='sample')
            pred_bench=predict_uploaded(models,uk_raw,seed,la_mode)
            if not pred_bench.empty and {'Lat','Lon'}.issubset(pred_bench):
                pred_bench=attach_crust1_reference(pred_bench)
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

                if 'H_GAME_LuffiDucea2022_km' in pred_bench:
                    with st.expander('GAME diagnostics',expanded=False):
                        st.caption('Luffi & Ducea (2022) GAME mohometers are reconstructed from the published T2 calibration table using local LOWESS-style interpolation, then combined with the paper-style MAD filtering into a consensus Moho estimate.')
                        uk_gtab1,uk_gtab2,uk_gtab3=st.tabs(['Consensus','Reliability','Calibration'])
                        with uk_gtab1:
                            uk_gx1,uk_gx2=st.columns([1,0.35])
                            uk_game_point_size=typed_slider(uk_gx2,'Point size',2,12,5,1,key='uk_game_diag_point_size')
                            uk_gfig=game_consensus_figure(pred_bench,'Predicted_km',uk_game_point_size)
                            if uk_gfig is not None:
                                st.plotly_chart(uk_gfig,width='stretch',key='uk_game_consensus_fig')
                            uk_status_counts=pred_bench['GAME_Luffi2022_Status'].fillna('unknown').value_counts().rename_axis('Status').reset_index(name='Rows') if 'GAME_Luffi2022_Status' in pred_bench else pd.DataFrame()
                            if not uk_status_counts.empty:
                                table_action_card('GAME status counts',uk_status_counts,'uk_game_status_counts.csv','uk_game_status_counts')
                        with uk_gtab2:
                            st.caption('The original GAME app explicitly tracks how many mohometers survive data-availability, reference-model residual/RMSE, and STD/MAD filtering. Low kept N, high MAD/IQR, wide bootstrap CI, or an "all valid; high spread" status should be treated as lower-confidence interpretation.')
                            uk_rx1,uk_rx2=st.columns([1,0.35])
                            uk_game_rel_size=typed_slider(uk_rx2,'Point size',2,12,6,1,key='uk_game_reliability_point_size')
                            uk_rfig=game_reliability_figure(pred_bench,'Predicted_km',uk_game_rel_size)
                            if uk_rfig is not None:
                                st.plotly_chart(uk_rfig,width='stretch',key='uk_game_reliability_fig')
                            uk_rel_counts=pred_bench['GAME_Luffi2022_Reliability'].fillna('unknown').value_counts().rename_axis('Reliability').reset_index(name='Rows') if 'GAME_Luffi2022_Reliability' in pred_bench else pd.DataFrame()
                            if not uk_rel_counts.empty:
                                table_action_card('GAME reliability counts',uk_rel_counts,'uk_game_reliability_counts.csv','uk_game_reliability_counts')
                            uk_rel_summary=game_reliability_summary(pred_bench,'Predicted_km')
                            if not uk_rel_summary.empty:
                                table_action_card('GAME reliability by N',uk_rel_summary,'uk_game_reliability_by_n.csv','uk_game_reliability_by_n')
                        with uk_gtab3:
                            uk_sensor_options=[s for s,_,_ in GAME_SENSORS if s in game_calibrators()]
                            if uk_sensor_options:
                                uk_sensor=st.selectbox('Mohometer calibration',uk_sensor_options,index=0,key='uk_game_calibration_sensor',format_func=lambda s: game_sensor_labels().get(s,s))
                                uk_cfig=game_calibration_figure(pred_bench,uk_sensor)
                                if uk_cfig is not None:
                                    st.plotly_chart(uk_cfig,width='stretch',key='uk_game_calibration_fig')
                            table_action_card('GAME mohometer summary',game_sensor_summary(),'uk_game_mohometer_summary.csv','uk_game_mohometer_summary')

                st.subheader('Proxy comparison')
                uk_proxy_options=proxy_value_options_available(pred_bench)
                uk_proxy_thickness_options=proxy_library_methods_available(pred_bench)
                if uk_proxy_options or uk_proxy_thickness_options:
                    uk_x_options=[f'Predicted_km::{m}' for m in pred_bench['Model'].dropna().astype(str).unique()] + [f'Proxy::{p}' for p in uk_proxy_thickness_options]
                    def uk_proxy_x_label(option):
                        if option.startswith('Predicted_km::'):
                            return model_thickness_label(option.split('::',1)[1])
                        if option.startswith('Proxy::'):
                            return PROXY_THICKNESS_LABELS.get(option.split('::',1)[1], option.split('::',1)[1]) + ' [Km]'
                        return option
                    uk_color_labels={'Age_Ma':'Age [Ma]','Rock_Type_Model':'Rock type','Tectonic_Setting':'Tectonic setting','Arc_or_Segment':'Arc/segment','Geologic_Domain':'Geological domain','Dataset':'Dataset'}
                    uk_proxy_mode=st.radio('Proxy plot mode',['Proxy value','Proxy thickness'],horizontal=True,key='uk_proxy_mode')
                    uk_pc1,uk_pc2,uk_pc3=st.columns([1,1,1])
                    normalize_widget_state('uk_proxy_x',uk_x_options)
                    uk_x_choice=uk_pc1.selectbox('X axis',uk_x_options,index=0,key='uk_proxy_x',format_func=uk_proxy_x_label)
                    uk_show_proxy_formulas=uk_pc2.checkbox('Show formulas on graph',False,key='uk_proxy_show_formulas')
                    uk_proxy_df=tidy_numbers(pred_bench)
                    uk_x_col='Predicted_km'
                    if uk_x_choice.startswith('Predicted_km::'):
                        uk_model_choice=uk_x_choice.split('::',1)[1]
                        uk_proxy_df=uk_proxy_df[uk_proxy_df['Model'].astype(str).eq(uk_model_choice)]
                    elif uk_x_choice.startswith('Proxy::'):
                        uk_x_col=uk_x_choice.split('::',1)[1]
                    uk_fig=None; uk_proxy_formulas={}
                    if uk_proxy_mode == 'Proxy value':
                        if not uk_proxy_options:
                            uk_fig=None
                        else:
                            with st.expander('Proxy value controls',expanded=True):
                                uk_pcat1,uk_pcat2,uk_pcat3,uk_pcat4=st.columns([1.4,1.1,1,0.45])
                                uk_proxy=uk_pcat1.selectbox('Y axis',uk_proxy_options,index=0,key='uk_proxy',format_func=proxy_value_label)
                                uk_proxy_color_options=[c for c in ['Age_Ma','Rock_Type_Model','Tectonic_Setting','Arc_or_Segment','Geologic_Domain','Dataset'] if c in pred_bench]
                                uk_proxy_color=uk_pcat2.selectbox('Colour by',uk_proxy_color_options,index=0,key='uk_proxy_color',format_func=lambda c: uk_color_labels.get(c,c)) if uk_proxy_color_options else ''
                                uk_proxy_point_size=typed_slider(uk_pcat3,'Point size',2,12,5,1,key='uk_proxy_value_point_size')
                                uk_pt1,uk_pt2,uk_pt3=st.columns(3)
                                uk_proxy_trend=uk_pt1.selectbox('Moving summary',['Median','Mean','Both','Off'],index=0,key='uk_proxy_value_trend')
                                uk_proxy_trend_bin=typed_slider(uk_pt2,'Summary bin width [Km]',2.0,20.0,5.0,1.0,key='uk_proxy_value_trend_bin')
                                uk_proxy_trend_min_n=typed_slider(uk_pt3,'Summary minimum N',1,50,5,1,key='uk_proxy_value_trend_min_n')
                                _uk_clip_opts={'None':0,'0.5%':0.5,'1%':1.0,'2%':2.0,'5%':5.0}
                                _uk_clip=_uk_clip_opts[st.selectbox('Clip outliers',list(_uk_clip_opts),index=0,key='uk_proxy_value_clip',help='Trim both axes by removing the top and bottom N% of data values.')]
                                _uk_px=pd.to_numeric(uk_proxy_df[uk_x_col],errors='coerce').dropna()
                                _uk_py=pd.to_numeric(uk_proxy_df[uk_proxy],errors='coerce').dropna() if uk_proxy in uk_proxy_df else pd.Series(dtype=float)
                                uk_proxy_x_range=[float(_uk_px.quantile(_uk_clip/100)),float(_uk_px.quantile(1-_uk_clip/100))] if _uk_clip and not _uk_px.empty else None
                                uk_proxy_y_range=[float(_uk_py.quantile(_uk_clip/100)),float(_uk_py.quantile(1-_uk_clip/100))] if _uk_clip and not _uk_py.empty else None
                            uk_fig,uk_proxy_formulas=validation_proxy_figure(uk_proxy_df,uk_proxy,uk_x_col,uk_proxy_color,uk_proxy_point_size,uk_show_proxy_formulas,uk_proxy_trend,uk_proxy_trend_bin,uk_proxy_trend_min_n,uk_proxy_x_range,uk_proxy_y_range)
                    else:
                        with st.expander('Proxy thickness controls',expanded=True):
                            uk_default_methods=[m for m in ['H_Sundell2021_Paired_km','H_GAME_LuffiDucea2022_km','H_Sundell2021_SrY_km','H_Sundell2021_LaYbN_km','H_Hu2017_Collisional_SrY_km','H_Profeta2015_SrY_km','H_Mantle2008_CeY_sample_km'] if m in uk_proxy_thickness_options]
                            uk_methods=proxy_library_multiselect(uk_proxy_thickness_options,default=uk_default_methods or uk_proxy_thickness_options[:4],key='uk_proxy_methods')
                            uk_th_pt1,uk_th_pt2,uk_th_pt3,uk_th_pt4=st.columns([0.45,1,1,1])
                            uk_proxy_thickness_point_size=typed_slider(uk_th_pt1,'Size',2,12,5,1,key='uk_proxy_thickness_point_size')
                            uk_proxy_thickness_trend=uk_th_pt2.selectbox('Moving summary',['Median','Mean','Both','Off'],index=0,key='uk_proxy_thickness_trend')
                            uk_proxy_thickness_trend_bin=typed_slider(uk_th_pt3,'Summary bin width [Km]',2.0,20.0,5.0,1.0,key='uk_proxy_thickness_trend_bin')
                            uk_proxy_thickness_trend_min_n=typed_slider(uk_th_pt4,'Minimum N',1,50,5,1,key='uk_proxy_thickness_trend_min_n')
                            _uk_tclip_opts={'None':0,'0.5%':0.5,'1%':1.0,'2%':2.0,'5%':5.0}
                            _uk_tclip=_uk_tclip_opts[st.selectbox('Clip outliers',list(_uk_tclip_opts),index=0,key='uk_thick_clip',help='Trim both axes by removing the top and bottom N% of data values.')]
                            _uk_tx=pd.to_numeric(uk_proxy_df[uk_x_col],errors='coerce').dropna()
                            _uk_ty_vals=[pd.to_numeric(uk_proxy_df[m],errors='coerce').dropna() for m in uk_methods if m in uk_proxy_df]
                            _uk_ty=pd.concat(_uk_ty_vals) if _uk_ty_vals else pd.Series(dtype=float)
                            uk_thick_x_range=[float(_uk_tx.quantile(_uk_tclip/100)),float(_uk_tx.quantile(1-_uk_tclip/100))] if _uk_tclip and not _uk_tx.empty else None
                            uk_thick_y_range=[float(_uk_ty.quantile(_uk_tclip/100)),float(_uk_ty.quantile(1-_uk_tclip/100))] if _uk_tclip and not _uk_ty.empty else None
                        uk_fig,uk_proxy_formulas=validation_proxy_thickness_figure(uk_proxy_df,uk_x_col,uk_methods,uk_proxy_thickness_point_size,uk_show_proxy_formulas,uk_proxy_thickness_trend,uk_proxy_thickness_trend_bin,uk_proxy_thickness_trend_min_n,uk_thick_x_range,uk_thick_y_range)
                    if uk_fig is not None:
                        st.plotly_chart(uk_fig,width='stretch',key='uk_proxy_chart')
                        if uk_proxy_formulas:
                            for _fname,_fform in uk_proxy_formulas.items():
                                st.caption(f'**{_fname}**: {_fform}')
                    else:
                        st.info('No complete proxy rows are available after filtering.')
                else:
                    st.info('Proxy comparison needs a recognised proxy value or proxy-thickness method in the uploaded data.')

                with st.expander('Local crustal-thickness estimate and grouping workflow',expanded=True):
                    st.caption('Estimate local crustal thickness for a target point, CRUST1.0 cell, sample, or arc segment from nearby geochemical samples in a combined spatial + temporal neighbourhood. Nearest-N, radius, and time window define the candidate population; they are not error bars.')
                    if {'Lat','Lon'}.issubset(pred_bench):
                        uk_domain_priority=[c for c in ['Arc_or_Segment','Geologic_Domain','Tectonic_Setting','Dataset','Rock_Type_Model','Geologic_Era','Geologic_Period','Geologic_Epoch'] if c in pred_bench]
                        uk_extra_domains=[c for c in pred_bench.columns if c not in uk_domain_priority and c not in ['Model','Algorithm'] and pred_bench[c].dtype == object and 1 < pred_bench[c].nunique(dropna=True) <= 30]
                        uk_local_source_df=pred_bench.copy()

                        st.markdown('**1. Target**')
                        uk_tg1,uk_tg2,uk_tg3,uk_tg4=st.columns(4)
                        uk_target_mode=uk_tg1.selectbox('Target type',['CRUST1.0 grid cell','Sample point','User-defined point','Arc segment'],index=0,key='uk_local_target_mode')
                        uk_user_lat=uk_tg2.number_input('Target latitude',-90.0,90.0,0.0,0.1,key='uk_local_user_lat',disabled=uk_target_mode!='User-defined point')
                        uk_user_lon=uk_tg3.number_input('Target longitude',-180.0,180.0,0.0,0.1,key='uk_local_user_lon',disabled=uk_target_mode!='User-defined point')
                        uk_user_age=uk_tg4.number_input('Target age [Ma]',0.0,4500.0,0.0,1.0,key='uk_local_user_age',disabled=uk_target_mode!='User-defined point')
                        uk_arc_group_options=uk_domain_priority + uk_extra_domains[:8]
                        uk_arc_col=st.selectbox('Arc/segment field',uk_arc_group_options,index=0,key='uk_local_arc_col',disabled=uk_target_mode!='Arc segment') if uk_arc_group_options else None

                        st.markdown('**2. Time window**')
                        uk_tw1,uk_tw2,uk_tw3=st.columns(3)
                        uk_time_mode=uk_tw1.selectbox('Time mode',['Modern benchmark: Age_Ma <= 5','Fixed time slice','Age bins','Ignore age'],index=0,key='uk_local_time_mode')
                        uk_local_time_window=typed_slider(uk_tw2,'Fixed slice width [Ma]',1.0,200.0,10.0,1.0,key='uk_local_time_window',disabled=uk_time_mode!='Fixed time slice')
                        uk_local_age_bin=typed_slider(uk_tw3,'Age bin width [Ma]',1.0,500.0,10.0,1.0,key='uk_local_age_bin_width',disabled=uk_time_mode!='Age bins')

                        st.markdown('**3. Spatial neighbourhood**')
                        uk_sp1,uk_sp2,uk_sp3,uk_sp4=st.columns(4)
                        uk_spatial_mode=uk_sp1.selectbox('Spatial mode',['Radius with minimum-N fallback','Radius window','Nearest N'],index=0,key='uk_local_spatial_mode')
                        uk_local_initial_radius=typed_slider(uk_sp2,'Initial radius [Km]',25,500,100,25,key='uk_local_initial_radius')
                        uk_local_max_radius=typed_slider(uk_sp3,'Maximum radius [Km]',int(uk_local_initial_radius),1000,max(250,int(uk_local_initial_radius)),25,key='uk_local_max_radius',disabled=uk_spatial_mode!='Radius with minimum-N fallback')
                        uk_local_nearest_n=typed_slider(uk_sp4,'Nearest N',3,100,30,1,key='uk_local_nearest_n',disabled=uk_spatial_mode!='Nearest N')
                        uk_sp5,uk_sp6,uk_sp7=st.columns(3)
                        uk_local_radius_step=typed_slider(uk_sp5,'Radius step [Km]',10,200,50,10,key='uk_local_radius_step',disabled=uk_spatial_mode!='Radius with minimum-N fallback')
                        uk_local_min_n=typed_slider(uk_sp6,'Minimum N',3,50,10,1,key='uk_local_min_n')
                        uk_local_preferred_n=typed_slider(uk_sp7,'Preferred N',int(uk_local_min_n),100,max(30,int(uk_local_min_n)),1,key='uk_local_preferred_n')

                        uk_domain_mode=st.radio('Domain radius option',['Manual radius','Upload geological domain','Sample distribution long axis'],horizontal=True,key='uk_local_domain_mode')
                        uk_local_domain=None; uk_radius_col=None; uk_uploaded_records=[]; uk_sample_group_col=None
                        if uk_domain_mode == 'Upload geological domain':
                            uk_domain_file=st.file_uploader('Upload geological domain polygon',type=['geojson','json','csv','xlsx','xls'],key='uk_local_domain_upload',help='GeoJSON polygons, or a table with Domain, Lat, Lon polygon vertices.')
                            uk_uploaded_records=read_domain_records(uk_domain_file) if uk_domain_file else []
                            if uk_uploaded_records:
                                uk_local_source_df=attach_uploaded_geological_domains(uk_local_source_df,uk_uploaded_records)
                                uk_local_domain='Geological_Domain'
                                uk_radius_col='Geological_Domain_Long_Axis_km'
                                table_action_card('Uploaded geological domains',tidy_numbers(pd.DataFrame([{k:v for k,v in r.items() if k != 'Polygon'} for r in uk_uploaded_records])),'uk_uploaded_geological_domains.csv','uk_uploaded_geological_domains')
                            else:
                                st.info('Upload a GeoJSON polygon file or a Domain/Lat/Lon vertex table to use polygon long-axis radii.')
                        elif uk_domain_mode == 'Sample distribution long axis':
                            uk_sample_group_options=['All samples'] + uk_domain_priority + uk_extra_domains[:8]
                            uk_sample_group=st.selectbox('Build sample-distribution domain from',uk_sample_group_options,index=0,key='uk_local_sample_distribution_group')
                            uk_sample_group_col=None if uk_sample_group == 'All samples' else uk_sample_group
                            uk_local_source_df=attach_sample_distribution_domains(uk_local_source_df,uk_sample_group_col)
                            uk_local_domain=None if uk_sample_group_col is None else 'Sample_Distribution_Domain'
                            uk_radius_col='Sample_Distribution_Long_Axis_km'
                            table_action_card('Sample-distribution domains',tidy_numbers(uk_local_source_df[['Sample_Distribution_Domain','Sample_Distribution_Long_Axis_km']].drop_duplicates().sort_values('Sample_Distribution_Domain')),'uk_sample_distribution_domains.csv','uk_sample_distribution_domains')

                        st.markdown('**4. Filters**')
                        uk_f1,uk_f2,uk_f3,uk_f4=st.columns(4)
                        uk_proxy_h_features_filt=[c for c in PROXY_THICKNESS_LABELS if c in uk_local_source_df]
                        uk_local_value_options=[c for c in ['Predicted_km'] + uk_proxy_h_features_filt if c in uk_local_source_df]
                        uk_local_value=uk_f1.selectbox('Geochemical H estimate',uk_local_value_options,index=0,key='uk_local_value',format_func=lambda c: {'Predicted_km':'model: crustal thickness [Km]'}.get(c,PROXY_THICKNESS_LABELS.get(c,c)))
                        uk_rock_filter_options=sorted(uk_local_source_df['Rock_Type_Model'].dropna().astype(str).unique()) if 'Rock_Type_Model' in uk_local_source_df else []
                        uk_selected_local_rocks=uk_f2.multiselect('Rock type',uk_rock_filter_options,default=uk_rock_filter_options,key='uk_local_rocks') if uk_rock_filter_options else None
                        uk_exclude_flagged=uk_f3.checkbox('Exclude flagged samples',True,key='uk_local_exclude_flagged')
                        uk_complete_guo=uk_f4.checkbox('Complete Guo rows only',False,key='uk_local_complete_guo')
                        uk_f5,uk_f6=st.columns(2)
                        uk_exclude_high_loi=uk_f5.checkbox('Exclude high LOI',False,key='uk_local_exclude_loi')
                        uk_loi_max=typed_slider(uk_f6,'Maximum LOI',0.0,20.0,5.0,0.5,key='uk_local_loi_max',disabled=not uk_exclude_high_loi)
                        uk_local_candidates=uk_local_source_df.copy()
                        if uk_exclude_flagged and 'Reliability_Flags' in uk_local_candidates:
                            uk_local_candidates=uk_local_candidates[uk_local_candidates['Reliability_Flags'].fillna('').astype(str).str.strip().isin(['','OK','ok','Ok'])]
                        if uk_exclude_high_loi and 'LOI' in uk_local_candidates:
                            uk_local_candidates=uk_local_candidates[pd.to_numeric(uk_local_candidates['LOI'],errors='coerce').le(uk_loi_max) | uk_local_candidates['LOI'].isna()]
                        if uk_complete_guo:
                            ok_complete,_=complete(uk_local_candidates,GUO_FEATURES)
                            uk_local_candidates=uk_local_candidates[ok_complete]

                        uk_map_selected_available=False
                        if {'Lat','Lon'}.issubset(uk_local_candidates):
                            with st.expander('Create grouping from map selection',expanded=False):
                                st.caption('Use the Plotly lasso or box select tools on the map, then include Map-selected group in the grouping comparison. The selected points are treated as one temporary domain.')
                                uk_selectable=uk_local_candidates.dropna(subset=['Lat','Lon']).reset_index(drop=True)
                                if uk_selectable.empty:
                                    st.info('No selectable Lat/Lon rows after filters.')
                                else:
                                    uk_sel_color_options=[c for c in [uk_local_value,'Predicted_km'] + [p for p in PROXY_THICKNESS_LABELS if p in uk_selectable] + ['Age_Ma','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb','MgO','SiO2','Rock_Type_Model','Geologic_Domain','Arc_or_Segment','Dataset'] if c in uk_selectable and (not pd.api.types.is_numeric_dtype(uk_selectable[c]) or pd.to_numeric(uk_selectable[c],errors='coerce').notna().any())]
                                    uk_sel_color_options=list(dict.fromkeys(uk_sel_color_options))
                                    uk_sel_color=st.selectbox('Map selection colour by',uk_sel_color_options,index=0,key='uk_local_group_select_map_color',format_func=local_option_label)
                                    uk_sel_hover=hover_cols(uk_selectable,['Sample_ID','Age_Ma','Dataset','Arc_or_Segment','Geologic_Domain','Rock_Type_Model','Model',uk_local_value,'Predicted_km','Sr_Y','La_Yb_N','Ce_Y','Dy_Yb'] + [p for p in PROXY_THICKNESS_LABELS if p in uk_selectable],uk_sel_color,'Lat','Lon')
                                    uk_sel_fig=px.scatter_geo(
                                        tidy_numbers(uk_selectable),lat='Lat',lon='Lon',color=uk_sel_color,
                                        hover_data=map_hover_data(uk_selectable,uk_sel_hover),
                                        projection='natural earth',template='plotly_white',
                                        labels={uk_sel_color:local_option_label(uk_sel_color)}
                                    )
                                    uk_sel_fig.update_traces(marker=dict(size=7,line=dict(color='black',width=0.5)),selector=dict(type='scattergeo'))
                                    uk_sel_fig.update_layout(height=420,dragmode='lasso',margin=dict(l=10,r=10,t=10,b=10))
                                    try:
                                        uk_sel_event=st.plotly_chart(uk_sel_fig,width='stretch',key='uk_local_group_select_map',on_select='rerun',selection_mode=['lasso','box'])
                                        uk_selected_idx=plotly_selected_indices(uk_sel_event)
                                    except TypeError:
                                        st.plotly_chart(uk_sel_fig,width='stretch',key='uk_group_select_map_fallback')
                                        uk_selected_idx=[]
                                        st.info('This Streamlit version does not expose map selections; use uploaded geological domains or auto grouping instead.')
                                    if uk_selected_idx:
                                        uk_selected_idx=[i for i in uk_selected_idx if i < len(uk_selectable)]
                                        uk_selected_ids=set(uk_selectable.iloc[uk_selected_idx].index)
                                        uk_selectable['Map_Selected_Group']='Outside selection'
                                        uk_selectable.loc[list(uk_selected_ids),'Map_Selected_Group']='Selected polygon'
                                        uk_local_candidates=uk_selectable
                                        uk_map_selected_available=True
                                        st.success(f'{len(uk_selected_idx)} samples selected for Map-selected group.')
                                    else:
                                        st.caption('No selected samples yet.')

                        st.markdown('**5. Grouping workflow**')
                        with st.expander('Grouping controls',expanded=True):
                            uk_preset_group_options=['All samples'] + [c for c in ['Geologic_Domain','Arc_or_Segment','Tectonic_Setting','Dataset','Rock_Type_Model','Geologic_Era','Geologic_Period','Geologic_Epoch','Geologic_Age_Label','Age_Ma'] if c in uk_local_candidates]
                            uk_preset_group_options += [c for c in uk_extra_domains[:8] if c in uk_local_candidates and c not in uk_preset_group_options]
                            uk_preset_group_options=list(dict.fromkeys(uk_preset_group_options))
                            uk_default_preset_group='Geologic_Domain' if 'Geologic_Domain' in uk_preset_group_options else 'Arc_or_Segment' if 'Arc_or_Segment' in uk_preset_group_options else 'Age_Ma' if 'Age_Ma' in uk_preset_group_options else 'All samples'
                            uk_pg1,uk_pg2,uk_pg3,uk_pg4=st.columns([1,0.75,0.8,0.8])
                            uk_preset_group_col=uk_pg1.selectbox(
                                'Preset group by',uk_preset_group_options,
                                index=uk_preset_group_options.index(uk_default_preset_group) if uk_default_preset_group in uk_preset_group_options else 0,
                                key='uk_local_preset_group_col',
                                format_func=lambda c: 'All samples' if c == 'All samples' else 'Belt / arc / segment' if c == 'Arc_or_Segment' else 'Age bins' if c == 'Age_Ma' else local_option_label(c),
                            )
                            uk_preset_split_distance=uk_pg2.checkbox('Sub-split by distance',True,key='uk_local_preset_split_distance')
                            uk_preset_segment_km=typed_slider(uk_pg3,'Max segment [Km]',10,1000,100,10,key='uk_local_preset_segment_km',disabled=not uk_preset_split_distance)
                            uk_preset_age_bin_width=typed_slider(uk_pg4,'Age bin [Ma]',1.0,500.0,50.0,1.0,key='uk_local_preset_age_bin_width',disabled=uk_preset_group_col!='Age_Ma')
                            uk_compare_options=['Preset grouping','Manual radius','Sample distribution long axis','Auto KMeans','Auto DBSCAN','Auto Agglomerative']
                            if uk_uploaded_records:
                                uk_compare_options.insert(1,'Uploaded geological domain')
                            if uk_map_selected_available:
                                uk_compare_options.insert(1,'Map-selected group')
                            uk_default_compare=['Preset grouping']
                            if uk_domain_mode in uk_compare_options and uk_domain_mode not in uk_default_compare:
                                uk_default_compare.append(uk_domain_mode)
                            if 'Auto KMeans' in uk_compare_options and 'Auto KMeans' not in uk_default_compare:
                                uk_default_compare.append('Auto KMeans')
                            uk_grouping_methods=st.multiselect('Grouping methods to compare',uk_compare_options,default=uk_default_compare,key='uk_local_grouping_methods')
                            uk_ac1,uk_ac2,uk_ac3,uk_ac4=st.columns(4)
                            uk_proxy_h_features=[c for c in PROXY_THICKNESS_LABELS if c in uk_local_candidates]
                            uk_auto_features_available=[c for c in ['Age_Ma',uk_local_value,'Predicted_km'] + uk_proxy_h_features + ['Sr_Y','La_Yb_N','Ce_Y','SiO2','MgO','La','Yb','Sr','Y'] if c in uk_local_candidates]
                            uk_auto_features_available=list(dict.fromkeys([c for c in uk_auto_features_available if not str(c).startswith('CRUST1') and c != 'Residual_km']))
                            uk_default_auto=[c for c in ['Age_Ma',uk_local_value,'Predicted_km','H_Sundell2021_Paired_km','H_Sundell2021_SrY_km','H_Sundell2021_LaYbN_km','Sr_Y','La_Yb_N'] if c in uk_auto_features_available]
                            uk_auto_features=uk_ac1.multiselect('Auto-domain features',uk_auto_features_available,default=list(dict.fromkeys(uk_default_auto)),key='uk_local_auto_features',format_func=lambda c: {'Predicted_km':'model: crustal thickness [Km]','Age_Ma':'Age [Ma]','Sr_Y':'Sr/Y','La_Yb_N':'La/Yb(N)','Ce_Y':'Ce/Y'}.get(c,PROXY_THICKNESS_LABELS.get(c,c)))
                            uk_auto_k_mode=uk_ac2.selectbox('Auto K mode',['Max range','Manual K'],index=0,key='uk_local_auto_k_mode')
                            uk_auto_k=typed_slider(uk_ac2,'Starting K',2,12,4,1,key='uk_local_auto_k')
                            uk_auto_eps=typed_slider(uk_ac3,'DBSCAN eps',0.2,3.0,0.9,0.1,key='uk_local_auto_eps')
                            uk_auto_min=typed_slider(uk_ac4,'Auto min samples',3,30,8,1,key='uk_local_auto_min')
                            uk_rk1,uk_rk2,uk_rk3=st.columns(3)
                            uk_max_group_range=typed_slider(uk_rk1,'Max group long-axis [Km]',50,2000,500,50,key='uk_local_auto_max_group_range',disabled=uk_auto_k_mode!='Max range')
                            uk_max_h_range=typed_slider(uk_rk2,'Max group H range [Km]',0.0,60.0,0.0,1.0,key='uk_local_auto_max_h_range',disabled=uk_auto_k_mode!='Max range',help='0 disables this constraint.')
                            uk_max_auto_k=typed_slider(uk_rk3,'Maximum K',2,30,12,1,key='uk_local_auto_max_k',disabled=uk_auto_k_mode!='Max range')
                            uk_aw1,uk_aw2,uk_aw3,uk_aw4=st.columns(4)
                            uk_spatial_weight=typed_slider(uk_aw1,'Spatial weight',0.0,5.0,1.0,0.25,key='uk_local_auto_spatial_weight')
                            uk_age_weight=typed_slider(uk_aw2,'Age weight',0.0,5.0,1.0,0.25,key='uk_local_auto_age_weight')
                            uk_thickness_weight=typed_slider(uk_aw3,'Thickness weight',0.0,5.0,1.0,0.25,key='uk_local_auto_thickness_weight')
                            uk_chemistry_weight=typed_slider(uk_aw4,'Chemistry weight',0.0,5.0,0.5,0.25,key='uk_local_auto_chemistry_weight')

                        uk_local_boot=typed_slider(st,'Bootstrap repeats',100,2000,500,100,key='uk_local_boot')
                        uk_local_frames=[]; uk_domain_summaries=[]; uk_grouping_map_frames=[]
                        for uk_gm in uk_grouping_methods:
                            uk_gm_source=uk_local_candidates.copy()
                            uk_gm_domain=None; uk_gm_radius_col=None
                            if uk_gm == 'Preset grouping':
                                uk_base_col=None if uk_preset_group_col == 'All samples' else uk_preset_group_col
                                uk_gm_source=attach_preset_group_domains(uk_gm_source,uk_base_col,max_segment_km=uk_preset_segment_km,split_by_distance=uk_preset_split_distance,age_bin_width=uk_preset_age_bin_width)
                                uk_gm_domain='Preset_Group_ID'; uk_gm_radius_col='Preset_Group_Segment_Long_Axis_km'
                                uk_domain_summaries.append(preset_group_summary(uk_gm_source))
                            elif uk_gm == 'Uploaded geological domain' and uk_uploaded_records:
                                uk_gm_source=attach_uploaded_geological_domains(uk_gm_source,uk_uploaded_records)
                                uk_gm_domain='Geological_Domain'; uk_gm_radius_col='Geological_Domain_Long_Axis_km'
                            elif uk_gm == 'Map-selected group' and 'Map_Selected_Group' in uk_gm_source:
                                uk_gm_domain='Map_Selected_Group'; uk_gm_radius_col=None
                            elif uk_gm == 'Sample distribution long axis':
                                uk_gm_source=attach_sample_distribution_domains(uk_gm_source,uk_sample_group_col)
                                uk_gm_domain=None if uk_sample_group_col is None else 'Sample_Distribution_Domain'
                                uk_gm_radius_col='Sample_Distribution_Long_Axis_km'
                            elif uk_gm.startswith('Auto '):
                                uk_auto_method=uk_gm.replace('Auto ','')
                                uk_gm_source,uk_gm_domain,uk_auto_summary=attach_auto_domains(
                                    uk_gm_source,tuple(uk_auto_features),method=uk_auto_method,n_clusters=uk_auto_k,eps=uk_auto_eps,
                                    min_samples=uk_auto_min,spatial_weight=uk_spatial_weight,age_weight=uk_age_weight,
                                    thickness_weight=uk_thickness_weight,chemistry_weight=uk_chemistry_weight,
                                    label=uk_gm,auto_k_by_range=uk_auto_k_mode=='Max range',
                                    max_group_range_km=uk_max_group_range,max_h_range_km=uk_max_h_range,
                                    h_col=uk_local_value if uk_local_value in uk_gm_source else 'Predicted_km',
                                    max_clusters=uk_max_auto_k
                                )
                                uk_domain_summaries.append(uk_auto_summary)
                            else:
                                uk_gm_domain=None; uk_gm_radius_col=None
                            if uk_gm_domain and uk_gm_domain in uk_gm_source and {'Lat','Lon'}.issubset(uk_gm_source):
                                uk_gm_map=uk_gm_source.dropna(subset=['Lat','Lon']).copy()
                                uk_gm_map['Grouping_Method']=uk_gm
                                uk_gm_map['Group_ID']=uk_gm_map[uk_gm_domain].astype(str)
                                uk_grouping_map_frames.append(uk_gm_map)
                            uk_gm_targets=build_local_targets(uk_target_mode,uk_gm_source,uk_user_lat,uk_user_lon,uk_user_age,uk_arc_col)
                            uk_gm_targets=attach_target_domains_from_samples(uk_gm_targets,uk_gm_source,uk_gm_domain,uk_gm_radius_col)
                            if uk_target_mode == 'CRUST1.0 grid cell' and uk_gm_targets.empty:
                                continue
                            uk_gm_df=local_crustal_thickness_estimates(
                                uk_gm_targets,uk_gm_source,uk_local_value,uk_time_mode,uk_user_age,uk_local_time_window,uk_local_age_bin,
                                uk_spatial_mode,uk_local_initial_radius,uk_local_max_radius,uk_local_radius_step,uk_local_nearest_n,
                                uk_local_min_n,uk_local_preferred_n,tuple(uk_selected_local_rocks) if uk_selected_local_rocks else None,uk_gm_domain,uk_gm_radius_col,uk_local_boot,seed
                            )
                            if not uk_gm_df.empty:
                                uk_gm_df.insert(0,'Grouping_Method',uk_gm)
                                uk_local_frames.append(uk_gm_df)
                        uk_local_df=pd.concat(uk_local_frames,ignore_index=True) if uk_local_frames else pd.DataFrame()
                        st.session_state['_uk_local_df']=uk_local_df.copy() if not uk_local_df.empty else pd.DataFrame()
                        if uk_domain_summaries:
                            uk_auto_summary_df=pd.concat([d for d in uk_domain_summaries if not d.empty],ignore_index=True) if any(not d.empty for d in uk_domain_summaries) else pd.DataFrame()
                            if not uk_auto_summary_df.empty:
                                table_action_card('Grouping summary',uk_auto_summary_df,'uk_grouping_summary.csv','uk_grouping_summary')
                                if 'K_Selection' in uk_auto_summary_df and uk_auto_summary_df['K_Selection'].astype(str).str.contains('not fully met',case=False,na=False).any():
                                    st.warning('Auto grouping reached the maximum K before every group met the selected range target. Increase Maximum K or relax the max range.')
                        if uk_grouping_map_frames:
                            uk_group_map=pd.concat(uk_grouping_map_frames,ignore_index=True)
                            st.session_state['_rs_group_map']=uk_group_map.copy()
                            st.session_state['_uk_group_map']=uk_group_map.copy()
                            st.session_state['_uk_local_df']=uk_local_df.copy() if not uk_local_df.empty else pd.DataFrame()
                            st.session_state['_uk_local_value']=uk_local_value
                            st.markdown('**6. Grouping diagnostics**')
                            _render_grouping_display('uk')
                        elif uk_grouping_methods:
                            st.info('Select at least one non-manual grouping method, such as Auto KMeans or Sample distribution long axis, to show the grouping map and proxy graph.')
                        if uk_target_mode == 'CRUST1.0 grid cell' and uk_local_df.empty:
                            st.info('CRUST1.0 targets need the default CRUST_1_0_excel.csv grid plus prediction Lat/Lon points.')
                        if uk_local_df.empty:
                            st.info('No local crustal-thickness estimates could be calculated. Check target coordinates, filters, Age_Ma, and selected model value availability.')
                        else:
                            st.markdown('**7. Estimate statistics**')
                            if {'Grouping_Method','Local_Domain_Value','Local_Median_H_km'}.issubset(uk_local_df):
                                _uk_grp_cols=[c for c in ['Grouping_Method','Local_Domain_Value','Model'] if c in uk_local_df]
                                _uk_grp_aggs={'Local_Median_H_km':['count','median']}
                                for _ac in ['Local_Mean_H_km','Bootstrap_95CI_Low_Median','Bootstrap_95CI_High_Median','Local_N','Local_IQR_H_km','Local_Radius_km','Residual_Local_vs_CRUST1_Total_km','Residual_Local_vs_CRUST1_Crystalline_km']:
                                    if _ac in uk_local_df:
                                        _uk_grp_aggs[_ac]='median'
                                if 'Good_Local_Estimate' in uk_local_df:
                                    _uk_grp_aggs['Good_Local_Estimate']='mean'
                                _uk_domain_summary=uk_local_df.groupby(_uk_grp_cols,dropna=False).agg(_uk_grp_aggs).reset_index()
                                _uk_domain_summary.columns=['_'.join([str(x) for x in c if x]) for c in _uk_domain_summary.columns]
                                _uk_rename={'Local_Median_H_km_count':'N_targets','Local_Median_H_km_median':'Median_H_km'}
                                _uk_domain_summary=_uk_domain_summary.rename(columns=_uk_rename)
                                table_action_card('Group-level crustal-thickness summary',tidy_numbers(_uk_domain_summary),'uk_local_group_summary.csv','uk_local_group_summary')
                                with st.expander('Individual target rows',expanded=False):
                                    uk_lead_cols=[c for c in ['Grouping_Method','Target_Type','Target_ID','Model','Sample_ID','Dataset','Arc_or_Segment','Geologic_Domain','Local_Domain_Column','Local_Domain_Value','Age_Ma','Lat','Lon','Local_Median_H_km','Local_Mean_H_km','Bootstrap_95CI_Low_Median','Bootstrap_95CI_High_Median','Local_N','Local_Radius_km','Local_Time_Window_Ma','Local_Radius_Source','Local_Domain_Long_Axis_km','CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Sediment_km','Residual_Local_vs_CRUST1_Total_km','Residual_Local_vs_CRUST1_Crystalline_km','Good_Local_Estimate','Low_N','Expanded_Radius','High_IQR','High_Bootstrap_Uncertainty','High_Sediment_Cover','No_Estimate'] if c in uk_local_df]
                                    uk_rest_cols=[c for c in uk_local_df.columns if c not in uk_lead_cols]
                                    table_action_card('Local crustal-thickness estimates',uk_local_df[uk_lead_cols+uk_rest_cols],'uk_local_crustal_thickness_estimates.csv','uk_local_estimate_statistics')
                            else:
                                uk_lead_cols=[c for c in ['Grouping_Method','Target_Type','Target_ID','Model','Sample_ID','Dataset','Arc_or_Segment','Geologic_Domain','Local_Domain_Column','Local_Domain_Value','Age_Ma','Lat','Lon','Local_Median_H_km','Local_Mean_H_km','Bootstrap_95CI_Low_Median','Bootstrap_95CI_High_Median','Local_N','Local_Radius_km','Local_Time_Window_Ma','Local_Radius_Source','Local_Domain_Long_Axis_km','CRUST1_Total_Crust_km','CRUST1_Crystalline_Crust_km','CRUST1_Sediment_km','Residual_Local_vs_CRUST1_Total_km','Residual_Local_vs_CRUST1_Crystalline_km','Good_Local_Estimate','Low_N','Expanded_Radius','High_IQR','High_Bootstrap_Uncertainty','High_Sediment_Cover','No_Estimate'] if c in uk_local_df]
                                uk_rest_cols=[c for c in uk_local_df.columns if c not in uk_lead_cols]
                                table_action_card('Local crustal-thickness estimates',uk_local_df[uk_lead_cols+uk_rest_cols],'uk_local_crustal_thickness_estimates.csv','uk_local_estimate_statistics')
                    else:
                        st.info('Grouping workflow needs Lat and Lon columns in the uploaded data.')

                st.caption('Download predictions in the Summary tab → Export results.')

with t_result_summary:
    st.header('Summary')
    _rs_pred=st.session_state.get('_rs_pred_bench',pd.DataFrame())
    _rs_group=st.session_state.get('_rs_group_map',pd.DataFrame())
    if _rs_pred.empty:
        st.info('Run predictions and (optionally) the grouping workflow in the Predict tab first.')
    else:
        _rs_has_groups=not _rs_group.empty and 'Group_ID' in _rs_group and 'Grouping_Method' in _rs_group
        # ── Row 1: data / model controls ─────────────────────────────────────
        _rs_c1,_rs_c2,_rs_c3,_rs_c4,_rs_c5=st.columns([1.3,1.2,1.1,0.9,0.6])
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
        _rs_point_size=typed_slider(_rs_c5,'Pt size',2,14,4,1,key='rs_point_size')
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
            _rs_grp_cols=[c for c in [_rs_x,'Model','Algorithm'] if c in _rs_vdf]
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
            if {'Lat','Lon'}.issubset(_rs_vdf.columns):
                st.divider()
                st.subheader('Prediction map')
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

            if _esh_summ and not _exp_main.empty:
                _grp_col = 'Group_ID' if 'Group_ID' in _exp_main else 'Model'
                _exp_grp_agg = (
                    _exp_main.groupby([_grp_col, 'Model'], dropna=False)
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
