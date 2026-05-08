"""Parked: auto-clustering helpers (KMeans / DBSCAN / Agglomerative + GMM
two-population detection) lifted out of Mohometer.py because the geological
interpretation was unreliable.

This module is intentionally NOT imported by Mohometer.py. To re-enable:
  1. Re-add `from engine._parked_auto_clustering import (
        attach_auto_domains, auto_labels_with_range, cluster_population_stats
     )` near the top of Mohometer.py.
  2. Restore the auto-clustering branches in the validate/predict grouping
     loops (look for the comment `# auto-clustering removed` for the spots).
  3. Restore the `cluster_diagnostic` parameter in
     local_crustal_thickness_estimates and a UI toggle to drive it.

Self-contained: imports only its own dependencies (sklearn) and ships its
own copies of the small geometric helpers it uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ─── Local copies of small geometric helpers (kept here so this module is
# usable without importing Mohometer.py) ─────────────────────────────────────

def _lonlat_xy_km(lon, lat, ref_lat=None):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    ref = np.nanmean(lat) if ref_lat is None else float(ref_lat)
    x = lon * 111.32 * np.cos(np.radians(ref))
    y = lat * 110.57
    return np.column_stack([x, y])


def _long_axis_km_from_lonlat(lon, lat):
    pts = _lonlat_xy_km(lon, lat)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return np.nan
    centered = pts - pts.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = centered @ vh[0]
        return float(np.nanmax(axis) - np.nanmin(axis))
    except Exception:
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        return float(np.nanmax(d))


def _tidy_numbers(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.select_dtypes(include='number').columns:
        out[c] = pd.to_numeric(out[c], errors='coerce').round(4)
    return out


# ─── Two-population detection (GMM) ──────────────────────────────────────────

def cluster_population_stats(values, target_value, min_cluster_n=5, seed=42):
    vals = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    empty = {
        'Thickness_Cluster_Count': np.nan,
        'Selected_Cluster_ID': np.nan,
        'Selected_Cluster_N': np.nan,
        'Selected_Cluster_Median': np.nan,
        'Selected_Cluster_Q25': np.nan,
        'Selected_Cluster_Q75': np.nan,
        'Selected_Cluster_Flag': 'not run',
    }
    if len(vals) < max(10, 2 * int(min_cluster_n)) or pd.isna(target_value):
        return empty
    x = vals.to_numpy(dtype=float).reshape(-1, 1)
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
        q25 = float(selected_vals.quantile(0.25))
        q75 = float(selected_vals.quantile(0.75))
        return {
            'Thickness_Cluster_Count': 2,
            'Selected_Cluster_ID': selected_id + 1,
            'Selected_Cluster_N': int(len(selected_vals)),
            'Selected_Cluster_Median': float(selected_vals.median()),
            'Selected_Cluster_Q25': q25,
            'Selected_Cluster_Q75': q75,
            'Selected_Cluster_Flag': 'two populations detected',
        }
    except Exception:
        return empty


# ─── Auto-K selection wrapper ────────────────────────────────────────────────

def auto_labels_with_range(method, X, source, n_clusters, min_samples=8,
                           max_group_range_km=np.nan, max_h_range_km=np.nan,
                           h_col='Predicted_km', max_clusters=12):
    method = method or 'KMeans'
    start_k = max(2, int(n_clusters))
    max_k = max(start_k, int(max_clusters))
    use_range = (
        (pd.notna(max_group_range_km) and float(max_group_range_km) > 0)
        or (pd.notna(max_h_range_km) and float(max_h_range_km) > 0)
    )
    methods_with_k = ['KMeans', 'Agglomerative']
    if method not in methods_with_k or not use_range:
        if method == 'Agglomerative':
            return AgglomerativeClustering(n_clusters=start_k).fit_predict(X), start_k, 'manual K'
        return KMeans(n_clusters=start_k, random_state=42, n_init=20).fit_predict(X), start_k, 'manual K'

    best_labels = None
    best_k = start_k
    best_reason = 'max K used; range target not fully met'
    for k in range(start_k, max_k + 1):
        labels = (AgglomerativeClustering(n_clusters=k).fit_predict(X)
                  if method == 'Agglomerative'
                  else KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X))
        ok = True
        for label_id in np.unique(labels):
            g = source.iloc[np.where(labels == label_id)[0]]
            if len(g) < int(min_samples):
                ok = False
                break
            if pd.notna(max_group_range_km) and float(max_group_range_km) > 0 and {'Lat', 'Lon'}.issubset(g):
                axis = _long_axis_km_from_lonlat(g['Lon'], g['Lat'])
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


# ─── Attach an auto-clustered domain to a sample dataframe ──────────────────

def attach_auto_domains(df, feature_cols, method='KMeans', n_clusters=4,
                        eps=0.85, min_samples=8, spatial_weight=1.0,
                        age_weight=1.0, thickness_weight=1.0,
                        chemistry_weight=0.5, label='Auto domain',
                        auto_k_by_range=False, max_group_range_km=np.nan,
                        max_h_range_km=np.nan, h_col='Predicted_km',
                        max_clusters=12):
    out = df.copy()
    out_col = label.replace(' ', '_').replace('-', '_') + '_ID'
    out[out_col] = pd.Series([pd.NA] * len(out), index=out.index, dtype='object')
    if out.empty or not {'Lat', 'Lon'}.issubset(out):
        return out, out_col, pd.DataFrame()

    cols = []
    weights = []
    if {'Lat', 'Lon'}.issubset(out):
        xy = pd.DataFrame(
            _lonlat_xy_km(pd.to_numeric(out['Lon'], errors='coerce'),
                          pd.to_numeric(out['Lat'], errors='coerce')),
            columns=['Auto_X_km', 'Auto_Y_km'], index=out.index,
        )
        out = pd.concat([out, xy], axis=1)
        cols += ['Auto_X_km', 'Auto_Y_km']
        weights += [spatial_weight, spatial_weight]

    for c in feature_cols:
        if c in out and c not in cols:
            cols.append(c)
            if c == 'Age_Ma':
                weights.append(age_weight)
            elif c in ['Predicted_km', 'Observed_km', 'Residual_km'] or str(c).endswith('_km'):
                weights.append(thickness_weight)
            else:
                weights.append(chemistry_weight)

    work = out[cols].apply(pd.to_numeric, errors='coerce')
    keep_cols = [c for c in work.columns if work[c].notna().sum() >= max(3, int(min_samples))]
    work = work[keep_cols]
    weights = [w for c, w in zip(cols, weights) if c in keep_cols]
    if work.empty:
        return out, out_col, pd.DataFrame()
    ok = work.notna().any(axis=1)
    if ok.sum() < max(3, int(min_samples)):
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
                h_col, max_clusters,
            )
        else:
            labels, chosen_k, k_reason = auto_labels_with_range(
                'KMeans', X, out.loc[work.index], n_clusters, min_samples,
                max_group_range_km if auto_k_by_range else np.nan,
                max_h_range_km if auto_k_by_range else np.nan,
                h_col, max_clusters,
            )
        out.loc[work.index, out_col] = [
            f'{label} {int(v) + 1}' if int(v) >= 0 else f'{label} outlier'
            for v in labels
        ]
    except Exception:
        return out, out_col, pd.DataFrame()

    summary = []
    for dom, g in out.dropna(subset=[out_col]).groupby(out_col, dropna=False):
        summary.append({
            'Grouping_Method': label,
            'Auto_Domain_ID': dom,
            'Chosen_K': chosen_k,
            'K_Selection': k_reason,
            'n': len(g),
            'Median_H_km': pd.to_numeric(g.get('Predicted_km'), errors='coerce').median()
                if 'Predicted_km' in g else np.nan,
            'Range_H_km': (pd.to_numeric(g.get(h_col), errors='coerce').max()
                           - pd.to_numeric(g.get(h_col), errors='coerce').min())
                if h_col in g else np.nan,
            'Median_Age_Ma': pd.to_numeric(g.get('Age_Ma'), errors='coerce').median()
                if 'Age_Ma' in g else np.nan,
            'Long_Axis_km': _long_axis_km_from_lonlat(g['Lon'], g['Lat'])
                if {'Lon', 'Lat'}.issubset(g) and len(g.dropna(subset=['Lon', 'Lat'])) >= 2
                else np.nan,
        })
    return out, out_col, _tidy_numbers(pd.DataFrame(summary))
