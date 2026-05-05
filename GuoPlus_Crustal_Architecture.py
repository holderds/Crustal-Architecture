# GuoPlus Crustal Architecture App
# Single-file Streamlit app built on Guo ExtraTrees workflow.

from __future__ import annotations
from pathlib import Path
from io import BytesIO
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

APP_VERSION = 'v10.1 feature-set'
GUO_FEATURES = ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Sr','Y','Rb','Ba','Hf','Nb','Ta','Th']
FEATURE_SETS = {
    'Full_32': GUO_FEATURES,
    'Top_24': ['Rb','Yb','Lu','Tm','Th','Ho','Y','Er','K2O','Ba','Sr','Al2O3','Pr','P2O5','Nd','Dy','Ce','La','MnO','TiO2','Na2O','Nb','Sm','Tb'],
    'Top_20': ['Rb','Yb','Lu','Tm','Th','Ho','Y','Er','K2O','Ba','Sr','Al2O3','Pr','P2O5','Nd','Dy','Ce','La','MnO','TiO2'],
    'Core_proxy_plus_major_context': ['SiO2','Al2O3','FeO','MnO','MgO','CaO','K2O','Sr','Y','La','Ce','Nd','Sm','Eu','Gd','Dy','Yb','Rb','Ba','Th'],
    'Tiny_proxy_set': ['Sr','Y','La','Yb','Ce','Dy','Gd','Rb','K2O','Al2O3'],
    'Majors_only': ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5'],
    'REE_only': ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'],
}
TRAINING_COLUMNS = ['Lon','Lat','Crust_Thickness','Age_Ma'] + GUO_FEATURES
CHON = {'La':0.237,'Sm':0.153,'Eu':0.058,'Gd':0.2055,'Yb':0.161}
FE2O3_TO_FEO = 0.8998

ALIASES = {
 'sample':'Sample_ID','sampleid':'Sample_ID','sample_id':'Sample_ID','sample name':'Sample_ID','sample_name':'Sample_ID','id':'Sample_ID',
 'longitude':'Lon','long':'Lon','lon':'Lon','longitude e':'Lon','latitude':'Lat','lat':'Lat','latitude n':'Lat',
 'age':'Age_Ma','age ma':'Age_Ma','age_ma':'Age_Ma','arc':'Arc_or_Segment','segment':'Arc_or_Segment','region':'Arc_or_Segment','belt':'Arc_or_Segment',
 'rock':'Rock_Type','rock_type':'Rock_Type','lithology':'Rock_Type','crustal thickness':'Crust_Thickness','crust thickness':'Crust_Thickness','moho':'Crust_Thickness','moho depth':'Crust_Thickness',
 'sio2':'SiO2','tio2':'TiO2','al2o3':'Al2O3','feo':'FeO','feot':'FeO','tfeo':'FeO','totalfeo':'FeO','total feo':'FeO','fe2o3':'Fe2O3','fe2o3t':'Fe2O3T','totalfe2o3':'Fe2O3T','total fe2o3':'Fe2O3T',
 'mno':'MnO','mgo':'MgO','cao':'CaO','ca0':'CaO','na2o':'Na2O','k2o':'K2O','p2o5':'P2O5','loi':'LOI',
 'la':'La','ce':'Ce','pr':'Pr','nd':'Nd','sm':'Sm','eu':'Eu','gd':'Gd','tb':'Tb','dy':'Dy','ho':'Ho','er':'Er','tm':'Tm','yb':'Yb','lu':'Lu','sr':'Sr','y':'Y','rb':'Rb','ba':'Ba','hf':'Hf','nb':'Nb','ta':'Ta','th':'Th','u':'U','zr':'Zr'
}
NUMERIC = list(dict.fromkeys(GUO_FEATURES + ['Fe2O3','Fe2O3T','U','Zr','LOI','Lat','Lon','Age_Ma','Crust_Thickness']))

def key(c):
    s = str(c).strip().lower()
    for ch in ['(',')','[',']','°','′','″']:
        s = s.replace(ch,' ')
    s = s.replace('-',' ').replace('_',' ').replace('/',' ')
    s = ' '.join(s.split())
    return s

def std_cols(df):
    out = df.copy(); ren = {}; used = set()
    for c in out.columns:
        k = key(c)
        cands = [k,k.replace(' ',''),k.replace(' ppm',''),k.replace(' wt%',''),k.replace(' wt pct',''),k.replace(' pct','')]
        parts = k.split()
        if parts: cands += [parts[0], parts[-1]]
        for x in cands:
            if x in ALIASES:
                t = ALIASES[x]
                if t not in used:
                    ren[c] = t; used.add(t)
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
    if 'Lat' in out: out['Lat'] = out['Lat'].apply(dms_to_float)
    if 'Lon' in out: out['Lon'] = out['Lon'].apply(dms_to_float)
    for c in NUMERIC:
        if c in out and c not in ['Lat','Lon']:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

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
    hits = sum(1 for f in ['sio2','tio2','al2o3','mgo','la','ce','nd','sr','y','th'] if f in vals)
    return hits >= 6 and ('crustal thickness' in joined or 'crust thickness' in joined or 'moho' in joined)

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

def find_training():
    for p in ['Table S1(1).xlsx','Table S1.xlsx','CrustThickness_5Ma_Tibet_Normalized.csv','data/training/Table S1(1).xlsx']:
        if Path(p).exists(): return Path(p)
    return None

def target_col(df):
    for c in ['Crust_Thickness','Crustal thickness','Crustal Thickness']:
        if c in df: return c
    for c in df.columns:
        if 'crust' in str(c).lower() and 'thick' in str(c).lower(): return c
    return None

@st.cache_resource(show_spinner=False)
def train_model(df, target, features, seed=42):
    if target is None:
        raise ValueError('Could not find a crustal thickness target column in the training table.')
    missing = [c for c in features if c not in df]
    if missing:
        raise ValueError(f'Training table is missing columns for this feature set: {", ".join(missing)}')
    clean = df[features+[target]].apply(pd.to_numeric,errors='coerce').dropna()
    if clean.empty:
        raise ValueError('No complete training rows remain after numeric coercion for this feature set.')
    model = Pipeline([('scaler',StandardScaler()),('etr',ExtraTreesRegressor(n_estimators=500,max_features=1.0,random_state=seed,n_jobs=-1))])
    model.fit(clean[features], clean[target].values.ravel())
    return model, clean

def cv(clean,target,features,seed=42):
    if len(clean) < 2:
        return clean[target].values.ravel(), clean[target].values.ravel(), np.nan, np.nan
    X = clean[features].values; y = clean[target].values.ravel(); pred = np.zeros_like(y,dtype=float)
    kf = KFold(n_splits=min(10,len(clean)),shuffle=True,random_state=seed)
    for tr,te in kf.split(X):
        m = Pipeline([('scaler',StandardScaler()),('etr',ExtraTreesRegressor(n_estimators=500,max_features=1.0,random_state=seed,n_jobs=-1))])
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

def div(a,b):
    a = pd.to_numeric(a,errors='coerce'); b = pd.to_numeric(b,errors='coerce')
    return np.where((a.notna())&(b.notna())&(b!=0),a/b,np.nan)

def pln(s):
    s = pd.to_numeric(s,errors='coerce'); return np.where(s>0,np.log(s),np.nan)

def enrich(df,la_yb_mode='raw_ppm'):
    out = df.copy(); sio2 = out['SiO2'] if 'SiO2' in out else pd.Series(np.nan,index=out.index); mgo = out['MgO'] if 'MgO' in out else pd.Series(np.nan,index=out.index)
    out['Rock_Type_Model'] = 'unclassified'; out.loc[(sio2>=44)&(sio2<=53)&(mgo>4),'Rock_Type_Model']='mafic'; out.loc[(sio2>=55)&(sio2<=68),'Rock_Type_Model']='intermediate'; out.loc[sio2>68,'Rock_Type_Model']='felsic'
    if {'Sr','Y'}.issubset(out): out['Sr_Y']=div(out['Sr'],out['Y'])
    if {'La','Yb'}.issubset(out):
        out['La_Yb_raw']=div(out['La'],out['Yb']); out['La_Yb_N']=out['La_Yb_raw'] if la_yb_mode=='already_normalized' else out['La_Yb_raw']/(CHON['La']/CHON['Yb'])
    if {'Ce','Y'}.issubset(out): out['Ce_Y']=div(out['Ce'],out['Y'])
    if {'MnO','MgO'}.issubset(out): out['MnO_MgO']=div(out['MnO'],out['MgO'])
    if {'Dy','Yb'}.issubset(out): out['Dy_Yb']=div(out['Dy'],out['Yb'])
    if {'Gd','Yb'}.issubset(out): out['Gd_Yb']=div(out['Gd'],out['Yb'])
    if {'Sm','Eu','Gd'}.issubset(out): out['Eu_Eu_star']=(out['Eu']/CHON['Eu'])/np.sqrt((out['Sm']/CHON['Sm'])*(out['Gd']/CHON['Gd']))
    if {'Rb','Sr'}.issubset(out): out['Rb_Sr']=div(out['Rb'],out['Sr'])
    if 'Sr_Y' in out: out['H_Profeta2015_SrY_km']=np.where(out['Sr_Y']>0,np.log(out['Sr_Y']/0.98)/0.047,np.nan); out['H_Sundell2021_SrY_km']=19.6*pln(out['Sr_Y'])-24; out['H_Zou2021_SrY_SVRE_km']=1.11*out['Sr_Y']+8.05
    if 'La_Yb_N' in out: out['H_Profeta2015_LaYbN_km']=(out['La_Yb_N']+7.25)/0.90; out['H_Sundell2021_LaYbN_km']=17*pln(out['La_Yb_N'])+6.9; out['H_Zou2021_LaYbN_SVRE_km']=21.277*np.log(1.0204*out['La_Yb_N'])
    if {'Sr_Y','La_Yb_N'}.issubset(out): out['H_Sundell2021_Paired_km']=10.3*pln(out['Sr_Y'])+8.8*pln(out['La_Yb_N'])-10.6
    if 'Ce_Y' in out: out['H_Mantle2008_CeY_sample_km']=np.where(out['Ce_Y']>0,np.log(out['Ce_Y']/0.3029)/0.0554,np.nan); out['H_Zou2021_CeY_SVRE_km']=18.0505*np.log(out['Ce_Y']+52.2238)
    out['Preferred_H_km']=out['H_Guo_ERT_km'] if 'H_Guo_ERT_km' in out else np.nan; out['Preferred_Method']=np.where(pd.notna(out['Preferred_H_km']),'Guo ERT ML','none')
    if 'H_Sundell2021_Paired_km' in out:
        m = pd.isna(out['Preferred_H_km']); out.loc[m,'Preferred_H_km']=out.loc[m,'H_Sundell2021_Paired_km']; out.loc[m,'Preferred_Method']='Sundell paired'
    flags=[]
    for _,r in out.iterrows():
        f=[]
        if r.get('Rock_Type_Model')=='mafic': f.append('Mafic: intermediate Sr/Y-La/Yb proxies caution')
        if pd.notna(r.get('LOI',np.nan)) and r.get('LOI')>3: f.append('High LOI alteration caution')
        if pd.notna(r.get('Sr_Y',np.nan)) and r.get('Sr_Y')>80 and pd.notna(r.get('Sr',np.nan)) and r.get('Sr')>1000: f.append('High Sr/Y + high Sr: possible cumulate/high-Sr effect')
        flags.append('; '.join(f) if f else 'OK')
    out['Reliability_Flags']=flags
    return out

def curves(proxy):
    H=np.linspace(5,90,300); d={}
    if proxy=='Sr_Y': d={'Profeta 2015':0.98*np.exp(0.047*H),'Sundell 2021':np.exp((H+24)/19.6),'Zou 2021':(H-8.05)/1.11}
    if proxy=='La_Yb_N': d={'Profeta 2015':0.90*H-7.25,'Sundell 2021':np.exp((H-6.9)/17),'Zou 2021':np.exp(H/21.277)/1.0204}
    if proxy=='Ce_Y': d={'Mantle & Collins 2008':0.3029*np.exp(0.0554*H),'M&C +3 km':0.3029*np.exp(0.0554*(H+3)),'M&C -3 km':0.3029*np.exp(0.0554*(H-3)),'Zou 2021':np.exp(H/18.0505)-52.2238}
    return H,d

def xlsx_bytes(df):
    bio=BytesIO()
    with pd.ExcelWriter(bio,engine='xlsxwriter') as w: df.to_excel(w,index=False,sheet_name='Results')
    return bio.getvalue()

st.set_page_config(page_title='GuoPlus Crustal Architecture',layout='wide')
st.title('GuoPlus Crustal Architecture')
st.caption(f'{APP_VERSION}: Guo ExtraTrees ML + reduced feature-set options + proxy curves + map + Fe2O3 to FeO conversion')
with st.sidebar:
    feature_set_name=st.selectbox('Guo feature set',list(FEATURE_SETS.keys()),index=0)
    selected_features=FEATURE_SETS[feature_set_name]
    st.caption(f'{len(selected_features)} model inputs')
    override=st.checkbox('Override default Guo model / retrain',False)
    la_mode=st.radio('La/Yb treatment',['raw_ppm','already_normalized'])
    pred_no_header=st.checkbox('Prediction file is Guo numeric no-header format',False)
    seed=st.number_input('Random state',0,value=42)

model=None; train_df=pd.DataFrame(); clean=pd.DataFrame(); target=None
try:
    if override:
        mode=st.sidebar.radio('Model source',['Upload training table','Upload joblib'])
        if mode=='Upload joblib':
            up=st.sidebar.file_uploader('Upload joblib',type=['joblib'])
            if up:
                tmp=Path(tempfile.gettempdir())/'guoplus_model.joblib'; tmp.write_bytes(up.getvalue()); obj=joblib.load(tmp)
                if isinstance(obj,dict) and 'model' in obj:
                    model=obj['model']; selected_features=obj.get('features',selected_features); feature_set_name=obj.get('feature_set',feature_set_name)
                else:
                    model=obj
        else:
            up=st.sidebar.file_uploader('Upload Guo training table',type=['csv','xlsx','xls'])
            if up: train_df=read_table(up,guo_no_header=True,expected=TRAINING_COLUMNS); target=target_col(train_df); model,clean=train_model(train_df,target,selected_features,seed)
    else:
        p=find_training()
        if p: train_df=read_table(p,guo_no_header=True,expected=TRAINING_COLUMNS); target=target_col(train_df); model,clean=train_model(train_df,target,selected_features,seed); st.success(f'Auto-trained {feature_set_name} Guo model from {p}')
        else: st.warning('No default training file found. Add Table S1(1).xlsx beside the app or use override.')
except Exception as e:
    st.error(f'Model setup failed: {e}')

if not clean.empty:
    with st.expander('Guo validation plot',expanded=True):
        y,p,r2,rmse=cv(clean,target,selected_features,seed); c1,c2=st.columns([1,2]); c1.metric('Feature set',feature_set_name); c1.metric('Training rows',len(clean)); c1.metric('R2',f'{r2:.3f}'); c1.metric('RMSE',f'{rmse:.1f} km')
        fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(y,p,25,color='r'); ax.plot([0,90],[0,90],'--',lw=2,color='b'); ax.plot([10,90],[0,80],'--',lw=2,color='g',alpha=.5); ax.plot([0,80],[10,90],'--',lw=2,color='g',alpha=.5); ax.set(xlabel='Observed',ylabel='Predicted',title='Crustal thickness',xlim=(0,90),ylim=(0,90)); ax.text(10,75,f'R2 = {r2:.3f}',fontsize=14); ax.text(10,70,f'RMSE = {rmse:.1f}',fontsize=14); c2.pyplot(fig)
        bio=BytesIO(); joblib.dump({'model':model,'features':selected_features,'feature_set':feature_set_name,'app_version':APP_VERSION},bio); st.download_button('Download trained Guo joblib',bio.getvalue(),f'guo_ert_{feature_set_name}.joblib','application/octet-stream')

uploaded=st.file_uploader('Upload prediction CSV/XLSX',type=['csv','xlsx','xls'])
if not uploaded: st.info('Upload a prediction dataset to begin.'); st.stop()
raw=read_table(uploaded,guo_no_header=pred_no_header,expected=GUO_FEATURES)
res=predict(model,raw,selected_features,feature_set_name) if model is not None else raw.copy(); res=enrich(res,la_mode)

t1,t2,t3,t4,t5=st.tabs(['QA','Predictions','Map','Proxy plots','Downloads'])
with t1:
    ok,miss=complete(res,selected_features); c1,c2,c3=st.columns(3); c1.metric('Rows',len(res)); c2.metric('Complete Guo rows',int(ok.sum())); c3.metric('Missing Guo features',len(miss));
    if miss: st.write(miss)
    st.dataframe(pd.DataFrame({'Column':res.columns,'Non_null':[int(res[c].notna().sum()) for c in res.columns],'Rows':len(res)}),use_container_width=True)
with t2:
    cols=[c for c in ['Sample_ID','Lat','Lon','Arc_or_Segment','Age_Ma','Rock_Type_Model','FeO','FeO_Source','H_Guo_ERT_km','Guo_ERT_Status','Sr_Y','La_Yb_N','Ce_Y','H_Sundell2021_Paired_km','Preferred_H_km','Preferred_Method','Reliability_Flags'] if c in res]
    st.dataframe(res[cols+[c for c in res.columns if c not in cols]],use_container_width=True)
with t3:
    if {'Lat','Lon'}.issubset(res):
        m=plot_df(res.dropna(subset=['Lat','Lon'])); color_options=[c for c in ['H_Guo_ERT_km','Preferred_H_km','Sr_Y','La_Yb_N','Ce_Y'] if c in m]
        if color_options:
            color=st.selectbox('Colour by',color_options)
            hover=hover_cols(m,['Sample_ID','Arc_or_Segment','Age_Ma','Rock_Type_Model','Guo_Feature_Set','Reliability_Flags'],color,'Lat','Lon')
            fig=px.scatter_geo(m,lat='Lat',lon='Lon',color=color,hover_name='Sample_ID' if 'Sample_ID' in m and 'Sample_ID' not in hover else None,hover_data=hover,projection='natural earth'); fig.update_layout(height=650); st.plotly_chart(fig,use_container_width=True)
        else:
            st.info('No plottable colour columns found.')
    else: st.info('No Lat/Lon columns found.')
with t4:
    plot_res=plot_df(res)
    nums=[c for c in plot_res.columns if pd.api.types.is_numeric_dtype(plot_res[c])]
    if len(nums) >= 2:
        x=st.selectbox('X',nums,index=nums.index('Preferred_H_km') if 'Preferred_H_km' in nums else 0); y=st.selectbox('Y',nums,index=nums.index('Sr_Y') if 'Sr_Y' in nums else min(1,len(nums)-1))
        color_col='Rock_Type_Model' if 'Rock_Type_Model' in plot_res else None
        hover=hover_cols(plot_res,['Sample_ID','Arc_or_Segment','Age_Ma','Guo_Feature_Set','Reliability_Flags'],x,y,color_col)
        st.plotly_chart(px.scatter(plot_res,x=x,y=y,color=color_col,hover_data=hover),use_container_width=True)
    else:
        st.info('Need at least two numeric columns for scatter plotting.')
    tx_options=[c for c in ['H_Guo_ERT_km','Preferred_H_km','Crust_Thickness'] if c in res]
    if tx_options:
        proxy=st.selectbox('Author curve proxy',['Sr_Y','La_Yb_N','Ce_Y']); tx=st.selectbox('Thickness axis',tx_options); H,cs=curves(proxy); sel=st.multiselect('Curves',list(cs.keys()),default=list(cs.keys()))
        fig=go.Figure(); d=res.dropna(subset=[tx,proxy]) if proxy in res and tx in res else pd.DataFrame(); fig.add_trace(go.Scatter(x=d[tx],y=d[proxy],mode='markers',name='Samples'))
        for n,v in cs.items():
            if n in sel: fig.add_trace(go.Scatter(x=H,y=v,mode='lines',name=n))
        fig.update_layout(xaxis_title=tx,yaxis_title=proxy,height=650,template='plotly_white'); st.plotly_chart(fig,use_container_width=True)
with t5:
    st.download_button('Download CSV',res.to_csv(index=False).encode('utf-8'),'guoplus_results.csv','text/csv')
    st.download_button('Download Excel',xlsx_bytes(res),'guoplus_results.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
