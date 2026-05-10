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
    Returns a string Series with NaN as 'unknown'. Preserves the input index
    so callers can concat with other label frames."""
    s = pd.Series(values) if not isinstance(values, pd.Series) else values
    out = pd.Series(['unknown'] * len(s), index=s.index, dtype='object')
    if len(breaks) < 2:
        return out
    nums = pd.to_numeric(s, errors='coerce')
    edges = np.asarray(breaks, dtype=float)
    # Extend the rightmost edge slightly so the maximum value falls in the last bin
    edges_use = edges.copy()
    edges_use[-1] = edges_use[-1] + 1e-9
    bin_idx = np.digitize(nums.to_numpy(dtype=float), edges_use, right=False) - 1
    nums_notna = nums.notna().to_numpy()
    for i in range(len(edges) - 1):
        mask = (bin_idx == i) & nums_notna
        if mask.any():
            # mask is positional; convert to label-based for .loc assignment
            out.iloc[np.where(mask)[0]] = label_for_bin(edges[i], edges[i + 1], decimals)
    return out


# ─── Filter specs and group resolution ───────────────────────────────────────

def candidate_group_columns(df: pd.DataFrame, max_unique: int = 200) -> list[str]:
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
    return out + extras


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
            # Rows with no value for this column pass through (not excluded by range filter).
            # This prevents a range filter on one column from wiping out rows that have
            # data for a different (categorical) partition column.
            if lo is not None:
                mask &= (ser >= float(lo)) | ser.isna()
            if hi is not None:
                mask &= (ser <= float(hi)) | ser.isna()
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
    # Stable group IDs (hash) so colours stay attached to a name across reruns.
    # combos.groupby(combos).groups returns label-based Index (matches work.index ⊆ out.index).
    for combo, label_idx in combos.groupby(combos).groups.items():
        gid = 'g_' + hashlib.md5(combo.encode('utf-8')).hexdigest()[:8]
        out.loc[label_idx, 'Group_ID'] = gid
        out.loc[label_idx, 'Group_Name'] = combo
        out.loc[label_idx, 'Group_Source'] = 'filter'
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


# ─── Polyline projection (Landing 3) ─────────────────────────────────────────

def _lonlat_to_km(lon, lat, ref_lat: float | None = None) -> np.ndarray:
    """Convert lon/lat (degrees) to a local equirectangular x/y projection in km.
    ref_lat fixes the longitudinal scale; if None, uses the mean latitude."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if ref_lat is None:
        finite_lat = lat[np.isfinite(lat)]
        ref_lat = float(np.nanmean(finite_lat)) if finite_lat.size else 0.0
    x = lon * 111.32 * np.cos(np.radians(ref_lat))
    y = lat * 110.57
    return np.column_stack([x, y])


def fit_pca_axis(lon, lat):
    """Fit a single straight-line axis through a cloud of (lon, lat) points
    using PCA on equirectangular-projected km coordinates.

    Returns (centroid_xy, direction_unit, axis_length_km, ref_lat) or None
    if there aren't at least 2 finite points.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    finite = np.isfinite(lon) & np.isfinite(lat)
    if finite.sum() < 2:
        return None
    ref_lat = float(np.nanmean(lat[finite]))
    xy = _lonlat_to_km(lon[finite], lat[finite], ref_lat=ref_lat)
    centroid = xy.mean(axis=0)
    centred = xy - centroid
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vh[0]
    # Normalise direction so the start (along=min) is the southern/western end —
    # makes "along-strike" labels intuitive across reruns.
    if direction[1] < 0 or (direction[1] == 0 and direction[0] < 0):
        direction = -direction
    along = centred @ direction
    length = float(np.nanmax(along) - np.nanmin(along))
    return centroid, direction, length, ref_lat


def project_onto_axis(lon, lat, centroid, direction, ref_lat):
    """Project samples onto a straight axis. Returns (along_km, across_km)
    with along zeroed at the southwestern end and across signed (positive
    on the left of the direction vector)."""
    xy = _lonlat_to_km(lon, lat, ref_lat=ref_lat)
    centred = xy - centroid
    along = centred @ direction
    perp = np.array([-direction[1], direction[0]])  # 90° CCW
    across = centred @ perp
    finite = np.isfinite(along)
    if finite.any():
        along = along - np.nanmin(along[finite])
    return along, across


def axis_endpoints_lonlat(centroid, direction, length_km, ref_lat) -> tuple:
    """Return ((lon0, lat0), (lon1, lat1)) for the two endpoints of the axis
    so it can be drawn as a line on a map."""
    half = length_km / 2.0
    p0 = centroid - direction * half
    p1 = centroid + direction * half
    cos_lat = np.cos(np.radians(ref_lat))
    if cos_lat == 0:
        cos_lat = 1.0
    lon0 = p0[0] / (111.32 * cos_lat)
    lat0 = p0[1] / 110.57
    lon1 = p1[0] / (111.32 * cos_lat)
    lat1 = p1[1] / 110.57
    return (float(lon0), float(lat0)), (float(lon1), float(lat1))


def attach_polyline_projections(df: pd.DataFrame, group_col: str = 'Group_ID',
                                 lon_col: str = 'Lon', lat_col: str = 'Lat'):
    """For each group, fit a PCA axis and project that group's samples onto it.
    Adds columns Along_Strike_km, Across_Strike_km to a copy of df.

    Returns (df_with_projections, group_axes) where group_axes is a dict
    {group_id: {'centroid', 'direction', 'length_km', 'ref_lat',
                'lon0','lat0','lon1','lat1'}} for plotting.
    """
    out = df.copy()
    out['Along_Strike_km'] = np.nan
    out['Across_Strike_km'] = np.nan
    group_axes: dict = {}
    if group_col not in out.columns or lon_col not in out.columns or lat_col not in out.columns:
        return out, group_axes
    for gid, g in out.dropna(subset=[group_col, lon_col, lat_col]).groupby(group_col):
        if len(g) < 2:
            continue
        fit = fit_pca_axis(g[lon_col].to_numpy(dtype=float),
                           g[lat_col].to_numpy(dtype=float))
        if fit is None:
            continue
        centroid, direction, length_km, ref_lat = fit
        along, across = project_onto_axis(
            g[lon_col].to_numpy(dtype=float),
            g[lat_col].to_numpy(dtype=float),
            centroid, direction, ref_lat,
        )
        out.loc[g.index, 'Along_Strike_km'] = np.round(along, 2)
        out.loc[g.index, 'Across_Strike_km'] = np.round(across, 2)
        (lon0, lat0), (lon1, lat1) = axis_endpoints_lonlat(centroid, direction, length_km, ref_lat)
        group_axes[gid] = {
            'centroid': centroid, 'direction': direction,
            'length_km': float(length_km), 'ref_lat': ref_lat,
            'lon0': lon0, 'lat0': lat0, 'lon1': lon1, 'lat1': lat1,
        }
    return out, group_axes


def rolling_window_smoothed(x, y, window_km: float, stat: str = 'median',
                             n_grid: int = 200):
    """Sliding-window aggregate of y vs x (both 1-D arrays).

    Returns (grid_x, smoothed_y, low_band, high_band) — low/high are Q25/Q75
    when stat='median' or mean ± 1 SD when stat='mean'. Empty arrays if there
    isn't enough data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    if len(x) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])
    grid = np.linspace(float(x.min()), float(x.max()), n_grid)
    half = float(window_km) / 2.0
    smoothed = np.full(n_grid, np.nan)
    lo = np.full(n_grid, np.nan)
    hi = np.full(n_grid, np.nan)
    for i, gx in enumerate(grid):
        mask = np.abs(x - gx) <= half
        n = int(mask.sum())
        if n < 3:
            continue
        ys = y[mask]
        if stat == 'mean':
            mu = float(np.mean(ys))
            sd = float(np.std(ys, ddof=1)) if n > 1 else 0.0
            smoothed[i] = mu; lo[i] = mu - sd; hi[i] = mu + sd
        else:
            smoothed[i] = float(np.median(ys))
            lo[i] = float(np.quantile(ys, 0.25))
            hi[i] = float(np.quantile(ys, 0.75))
    return grid, smoothed, lo, hi


# ─── Persistence helpers ─────────────────────────────────────────────────────

def round_trip_columns() -> list[str]:
    """Columns to write into exports / detect on re-upload to round-trip groups."""
    return ['Group_ID', 'Group_Name', 'Group_Source']


# ─── Group-then-calculate aggregation ────────────────────────────────────────

# Columns that are NOT aggregated as chemistry; they receive special treatment
_AGG_SKIP = frozenset({
    'Sample_ID', 'Group_ID', 'Group_Name', 'Group_Source',
    'Along_Strike_km', 'Across_Strike_km',
})

# Columns whose group mode (most-common value) is used instead of median
_AGG_META_CATS = (
    'Tectonic_Setting', 'Arc_or_Segment', 'Geologic_Domain', 'Dataset',
    'Rock_Type', 'Rock_Type_Model', 'Geologic_Era', 'Geologic_Period',
    'Geologic_Epoch', 'Geologic_Age_Label',
)


def aggregate_to_groups(
    df: pd.DataFrame,
    group_col: str = 'Group_ID',
    name_col: str = 'Group_Name',
    lat_col: str = 'Lat',
    lon_col: str = 'Lon',
    age_col: str = 'Age_Ma',
    target_cols: tuple = ('Crust_Thickness', 'Observed_km'),
    extra_agg_cols: tuple = (),
) -> pd.DataFrame:
    """Collapse a grouped DataFrame to one representative row per group.

    Rules
    -----
    * **Lat / Lon** → group median (spatial centroid).
    * **Age_Ma** → group median.
    * **Known-thickness target columns** (Crust_Thickness, Observed_km) → group median.
    * **Categorical metadata** (Tectonic_Setting, etc.) → mode (most common value).
    * **All other numeric columns** (chemistry) → group median.
    * **Group_ID / Group_Name / Group_Source** → passed through as-is from the
      first non-null value in the group.
    * **Along_Strike_km / Across_Strike_km** → dropped (per-sample projection).
    * **Sample_ID** → set to ``"<Group_Name> (n=<N>)"`` to indicate it is an
      aggregate.

    Parameters
    ----------
    df : DataFrame with a ``group_col`` column (and optionally ``name_col``).
    group_col : column that holds the group identifier.
    extra_agg_cols : additional column names to force-include as numeric medians.

    Returns
    -------
    One-row-per-group DataFrame, sorted by group_col, with a new column
    ``Agg_N`` recording the number of raw samples in each group.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if group_col not in df.columns:
        return pd.DataFrame()

    keep = df[df[group_col].notna()].copy()
    if keep.empty:
        return pd.DataFrame()

    meta_cat_set = set(_AGG_META_CATS)
    skip_set = set(_AGG_SKIP)

    # Determine column roles
    num_cols = [
        c for c in keep.columns
        if c not in skip_set
        and c not in meta_cat_set
        and c not in (lat_col, lon_col, age_col)
        and c not in set(target_cols)
        and pd.api.types.is_numeric_dtype(keep[c])
    ]

    rows = []
    for gid, grp in keep.groupby(group_col, sort=False):
        n = len(grp)
        gname = (
            grp[name_col].dropna().iloc[0]
            if name_col in grp.columns and grp[name_col].notna().any()
            else str(gid)
        )
        row: dict = {
            group_col: gid,
            name_col: gname,
            'Agg_N': n,
            'Sample_ID': f'{gname} (n={n})',
        }
        # Spatial centroid
        for _c in (lat_col, lon_col):
            if _c in grp.columns:
                v = pd.to_numeric(grp[_c], errors='coerce')
                row[_c] = float(v.median()) if v.notna().any() else np.nan
        # Age
        if age_col in grp.columns:
            v = pd.to_numeric(grp[age_col], errors='coerce')
            row[age_col] = float(v.median()) if v.notna().any() else np.nan
        # Target columns (Crust_Thickness, Observed_km …)
        for _c in target_cols:
            if _c in grp.columns:
                v = pd.to_numeric(grp[_c], errors='coerce')
                row[_c] = float(v.median()) if v.notna().any() else np.nan
        # Categorical metadata → mode
        for _c in _AGG_META_CATS:
            if _c in grp.columns:
                vals = grp[_c].dropna().astype(str)
                row[_c] = vals.mode().iloc[0] if not vals.empty else np.nan
        # Chemistry & other numerics → median
        for _c in list(num_cols) + list(extra_agg_cols):
            if _c in grp.columns:
                v = pd.to_numeric(grp[_c], errors='coerce')
                row[_c] = float(v.median()) if v.notna().any() else np.nan
        # Pass through Group_Source
        if 'Group_Source' in grp.columns:
            gs = grp['Group_Source'].dropna()
            row['Group_Source'] = str(gs.iloc[0]) if not gs.empty else 'aggregated'
        rows.append(row)

    out = pd.DataFrame(rows)
    # Restore column order from original (drop columns that weren't captured)
    orig_order = [c for c in df.columns if c in out.columns and c not in skip_set]
    extra = [c for c in out.columns if c not in set(orig_order)]
    out = out[list(dict.fromkeys(orig_order + extra))].reset_index(drop=True)
    return out
