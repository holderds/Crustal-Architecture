"""Rock type and age-bin classification."""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_rock_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Rock_Type_Model"] = "unclassified"
    if "SiO2" not in out.columns:
        return out
    sio2 = pd.to_numeric(out["SiO2"], errors="coerce")
    mgo = pd.to_numeric(out["MgO"], errors="coerce") if "MgO" in out.columns else np.nan
    mafic = (sio2 >= 44) & (sio2 <= 53) & (mgo > 4)
    transition = (sio2 > 53) & (sio2 < 55)
    intermediate = (sio2 >= 55) & (sio2 <= 68)
    felsic = sio2 > 68
    out.loc[mafic, "Rock_Type_Model"] = "mafic"
    out.loc[transition, "Rock_Type_Model"] = "basaltic_intermediate_transition"
    out.loc[intermediate, "Rock_Type_Model"] = "intermediate"
    out.loc[felsic, "Rock_Type_Model"] = "felsic"
    out["Is_Mafic"] = out["Rock_Type_Model"].eq("mafic")
    out["Is_Intermediate"] = out["Rock_Type_Model"].eq("intermediate")
    out["Is_Felsic"] = out["Rock_Type_Model"].eq("felsic")
    return out


def add_age_bins(df: pd.DataFrame, bin_size_ma: float = 10.0) -> pd.DataFrame:
    out = df.copy()
    if "Age_Ma" not in out.columns:
        out["Age_Bin"] = "unknown"
        return out
    age = pd.to_numeric(out["Age_Ma"], errors="coerce")
    low = (np.floor(age / bin_size_ma) * bin_size_ma).astype("Int64")
    high = (low + int(bin_size_ma)).astype("Int64")
    out["Age_Bin"] = np.where(age.notna(), low.astype(str) + "-" + high.astype(str) + " Ma", "unknown")
    return out
