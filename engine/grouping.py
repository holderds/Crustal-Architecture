"""Grouping helpers: numeric classification, filter application, group resolution.

Used by the Grouping tab in Mohometer.py. Self-contained — no Streamlit
dependencies in this module so it stays pure-pandas/numpy.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


# ─── Numeric classification ──────────────────────────────────────────────────

def equal_interval_breaks(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin edges across the value range (n_bins + 1 edges)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if lo == hi:
        return np.array([lo, hi + 1e-9])
    return np.linspace(lo, hi, n_bins + 1)


def geometric_interval_breaks(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Geometric-progression bin edges. Falls back to equal-interval if data
    span is non-positive (we can't take a log)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if lo <= 0 or lo == hi:
        # Shift to positive then geometric
        shift = (-lo) + 1.0 if lo <= 0 else 0.0
        if (lo + shift) == (hi + shift):
            return np.array([lo, hi + 1e-9])
        edges = np.geomspace(lo + shift, hi + shift, n_bins + 1) - shift
        return edges
    return np.geomspace(lo, hi, n_bins + 1)


def quantile_breaks(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile (equal-count) bin edges — every bin has roughly the same N."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(vals, qs))
    if len(edges) < 2:
        edges = np.array([float(np.nanmin(vals)), float(np.nanmax(vals)) + 1e-9])
    return edges


def jenks_breaks(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Jenks natural breaks. Uses jenkspy if available; otherwise falls back
    to a Fisher-Jenks approximation via mapclassify; otherwise quantile."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([])
    if len(vals) <= n_bins:
        return np.array(sorted(set(vals.tolist())) + [float(vals.max()) + 1e-9])
    try:
        import jenkspy
        breaks = jenkspy.jenks_breaks(vals, n_classes=n_bins)
        return np.asarray(breaks, dtype=float)
    except Exception:
        pass
    try:
        from mapclassify import NaturalBreaks
        nb = NaturalBreaks(vals, k=n_bins)
        edges = np.concatenate([[float(vals.min())], np.asarray(nb.bins, dtype=float)])
        return edges
    except Exception:
        # Fallback: quantile (won't minimise within-class variance like Jenks
        # but is a reasonable graceful degradation).
        return quantile_breaks(vals, n_bins)


def numeric_breaks(values, method: str, n_bins: int) -> np.ndarray:
    """Dispatcher for the four classification methods."""
    method = (method or 'equal').lower()
    n_bins = max(2, int(n_bins))
    if method == 'geometric':
        return geometric_interval_breaks(values, n_bins)
    if method == 'quantile':
        return quantile_breaks(values, n_bins)
    if method == 'jenks':
        return jenks_breaks(values, n_bins)
    return equal_interval_breaks(values, n_bins)


def label_for_bin(lo: float, hi: float, decimals: int = 1) -> str:
    """Human-readable bin label like '0.0–10.0'."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 'unknown'
    fmt = f'{{:.{decimals}f}}'
    return f'{fmt.format(lo)}–{fmt.format(hi)}'


def assign_numeric_bins(values, breaks: np.ndarray, decimals: int = 1) -> pd.Series:
    """Assign each value to a labelled bin using the given break edges.
    Returns a string Series with NaN as 'unknown'."""
    s = pd.Series(values).reset_index(drop=True)
    if len(breaks) < 2:
        return pd.Series(['unknown'] * len(s), index=s.index, dtype='object')
    nums = pd.to_numeric(s, errors='coerce')
    out = pd.Series(['unknown'] * len(s), index=s.index, dtype='object')
    edges = np.asarray(breaks, dtype=float)
    # Clamp to extend the rightmost edge slightly so the maximum value falls in the last bin
    edges_use = edges.copy()
    edges_use[-1] = edges_use[-1] + 1e-9
    bin_idx = np.digitize(nums.to_numpy(dtype=float), edges_use, right=False) - 1
    for i in range(len(edges) - 1):
        mask = (bin_idx == i) & nums.notna().to_numpy()
        if mask.any():
            out.loc[mask] = label_for_bin(edges[i], edges[i + 1], decimals)
    return out


# ─── Filter specs and group resolution ───────────────────────────────────────

def candidate_group_columns(df: pd.DataFrame, max_unique: int = 60) -> list[str]:
    """Columns that make sense as a categorical group key.
    Priority columns first, then any other low-cardinality string columns."""
    priority = [
        'Arc_or_Segment', 'Geologic_Domain', 'Tectonic_Setting', 'Country',
        'Dataset', 'Rock_Type_Model', 'Lithology_Type',
        'Geologic_Era', 'Geologic_Period', 'Geologic_Epoch', 'Geologic_Age_Label',
    ]
    out = [c for c in priority if c in df.columns]
    extras = []
    for c in df.columns:
        if c in out:
            continue
        if c in {'Sample_ID', 'Lat', 'Lon', 'Model', 'Algorithm', 'Group_ID', 'Group_Name'}:
            continue
        try:
            if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c]):
                n = df[c].nunique(dropna=True)
                if 1 < n <= max_unique:
                    extras.append(c)
        except Exception:
            continue
    return out + extras[:12]


def candidate_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns suitable for range filtering. Excludes Lat/Lon and IDs."""
    skip = {'Sample_ID', 'Lat', 'Lon', 'Group_ID', 'Group_Name'}
    out = []
    for c in df.columns:
        if c in skip:
            continue
        try:
            ser = pd.to_numeric(df[c], errors='coerce')
            if ser.notna().sum() >= 2 and float(ser.min()) != float(ser.max()):
                out.append(c)
        except Exception:
            continue
    # Put age + thickness near the top so they're easy to find
    priority = ['Age_Ma', 'Predicted_km', 'Observed_km',
                'CRUST1_Total_Crust_km', 'H_GAME_LuffiDucea2022_km',
                'SiO2', 'MgO', 'Sr_Y', 'La_Yb_N']
    head = [c for c in priority if c in out]
    tail = [c for c in out if c not in head]
    return head + tail


def apply_filter_specs(df: pd.DataFrame, specs: list[dict]) -> pd.Series:
    """Return boolean mask of rows that pass all filter specs (AND-combined)."""
    mask = pd.Series([True] * len(df), index=df.index)
    for spec in specs or []:
        col = spec.get('col')
        if not col or col not in df.columns:
            continue
        kind = spec.get('type')
        if kind == 'categorical':
            values = spec.get('values') or []
            if values:
                mask &= df[col].astype(str).isin([str(v) for v in values])
        elif kind == 'numeric_range':
            lo = spec.get('lo')
            hi = spec.get('hi')
            ser = pd.to_numeric(df[col], errors='coerce')
            if lo is not None:
                mask &= (ser >= float(lo)) | ser.isna()
            if hi is not None:
                mask &= (ser <= float(hi)) | ser.isna()
            mask &= ser.notna()
        elif kind == 'numeric_bin':
            method = spec.get('method', 'equal')
            n_bins = int(spec.get('n_bins', 4))
            keep_labels = spec.get('keep') or []
            ser = pd.to_numeric(df[col], errors='coerce')
            breaks = numeric_breaks(ser.dropna().to_numpy(), method, n_bins)
            labels = assign_numeric_bins(ser, breaks)
            if keep_labels:
                mask &= labels.isin(keep_labels)
            mask &= ser.notna()
    return mask


def auto_groups_from_filters(df: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Apply filters + partition matching rows by the categorical/binned values
    used in the filters. Each unique combination becomes one group.

    Returns a copy of df with extra columns: Group_ID, Group_Name, Group_Source.
    Rows that fail any filter get Group_ID = NaN.
    """
    out = df.copy().reset_index(drop=True)
    out['Group_ID'] = pd.Series([pd.NA] * len(out), dtype='object')
    out['Group_Name'] = pd.Series([pd.NA] * len(out), dtype='object')
    out['Group_Source'] = pd.Series([pd.NA] * len(out), dtype='object')
    if not specs:
        return out
    keep = apply_filter_specs(out, specs)
    work = out.loc[keep].copy()
    if work.empty:
        return out

    # Build the "partition keys" per spec
    key_frames = []
    key_names = []
    for spec in specs:
        col = spec.get('col')
        if not col or col not in work.columns:
            continue
        kind = spec.get('type')
        if kind == 'categorical':
            key_frames.append(work[col].astype(str))
            key_names.append(col)
        elif kind == 'numeric_bin':
            method = spec.get('method', 'equal')
            n_bins = int(spec.get('n_bins', 4))
            ser = pd.to_numeric(work[col], errors='coerce')
            breaks = numeric_breaks(ser.dropna().to_numpy(), method, n_bins)
            labels = assign_numeric_bins(ser, breaks)
            key_frames.append(labels.astype(str))
            key_names.append(f'{col} ({method})')
        # numeric_range filters constrain rows but don't subdivide groups
    if not key_frames:
        # Filters narrowed rows but didn't define a partition — single group
        out.loc[keep, 'Group_ID'] = 'group_1'
        out.loc[keep, 'Group_Name'] = 'filtered'
        out.loc[keep, 'Group_Source'] = 'filter'
        return out

    keys = pd.concat(key_frames, axis=1)
    keys.columns = key_names
    combos = keys.apply(lambda r: ' / '.join(str(v) for v in r.values), axis=1)
    # Stable group IDs (hash) so colours stay attached to a name across reruns
    for combo, idx in combos.groupby(combos).groups.items():
        gid = 'g_' + hashlib.md5(combo.encode('utf-8')).hexdigest()[:8]
        out.loc[work.index[idx], 'Group_ID'] = gid
        out.loc[work.index[idx], 'Group_Name'] = combo
        out.loc[work.index[idx], 'Group_Source'] = 'filter'
    return out


# ─── Per-group statistics ────────────────────────────────────────────────────

_DEFAULT_HOVER_COLS = [
    'Predicted_km', 'Observed_km', 'CRUST1_Total_Crust_km',
    'H_GAME_LuffiDucea2022_km',
    'H_Sundell2021_Paired_km', 'H_Sundell2021_SrY_km', 'H_Sundell2021_LaYbN_km',
    'H_Profeta2015_SrY_km', 'H_Profeta2015_LaYbN_km',
    'H_Mantle2008_CeY_sample_km', 'H_Zou2021_SrY_SVRE_km', 'H_Zou2021_LaYbN_SVRE_km',
    'Age_Ma', 'Sr_Y', 'La_Yb_N', 'SiO2', 'MgO',
]


def group_stats(df: pd.DataFrame, value_col: str = 'Predicted_km',
                hover_cols: Iterable[str] | None = None,
                decimals: int = 2) -> pd.DataFrame:
    """Per-group aggregates for the chosen value column plus hover context.

    Reports N, Mean, Median, MAD, Q25, Q75, Min, Max for value_col, plus
    the median of every hover col (proxies, GAME, etc.) so a tooltip can
    show all of it.
    """
    if 'Group_ID' not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    rows = []
    cols_to_add = list(hover_cols) if hover_cols is not None else _DEFAULT_HOVER_COLS
    cols_to_add = [c for c in cols_to_add if c in df.columns and c != value_col]
    for gid, g in df.dropna(subset=['Group_ID']).groupby('Group_ID', dropna=False):
        vals = pd.to_numeric(g[value_col], errors='coerce').dropna()
        n = int(len(vals))
        med = float(vals.median()) if n else np.nan
        mean = float(vals.mean()) if n else np.nan
        mad = float((vals - med).abs().median()) if n else np.nan
        row = {
            'Group_ID': gid,
            'Group_Name': g['Group_Name'].dropna().astype(str).iloc[0]
                if 'Group_Name' in g and not g['Group_Name'].dropna().empty else gid,
            'N': n,
            'Mean': mean,
            'Median': med,
            'MAD': mad,
            'Q25': float(vals.quantile(0.25)) if n else np.nan,
            'Q75': float(vals.quantile(0.75)) if n else np.nan,
            'Min': float(vals.min()) if n else np.nan,
            'Max': float(vals.max()) if n else np.nan,
        }
        for c in cols_to_add:
            cv = pd.to_numeric(g[c], errors='coerce').dropna()
            row[f'{c}_median'] = float(cv.median()) if not cv.empty else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        for c in out.select_dtypes(include='number').columns:
            out[c] = out[c].round(decimals)
    return out


# ─── Persistence helpers ─────────────────────────────────────────────────────

def round_trip_columns() -> list[str]:
    """Columns to write into exports / detect on re-upload to round-trip groups."""
    return ['Group_ID', 'Group_Name', 'Group_Source']
