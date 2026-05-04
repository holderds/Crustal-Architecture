"""Geochemical ratio calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

CHONDRITE = {"La": 0.237, "Sm": 0.153, "Eu": 0.058, "Gd": 0.2055, "Yb": 0.161}


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((a.notna()) & (b.notna()) & (b != 0), a / b, np.nan)


def add_ratios(df: pd.DataFrame, la_yb_mode: str = "raw_ppm") -> pd.DataFrame:
    out = df.copy()
    if {"Sr", "Y"}.issubset(out.columns): out["Sr_Y"] = safe_divide(out["Sr"], out["Y"])
    if {"La", "Yb"}.issubset(out.columns):
        out["La_Yb_raw"] = safe_divide(out["La"], out["Yb"])
        out["La_Yb_N"] = out["La_Yb_raw"] if la_yb_mode == "already_normalized" else out["La_Yb_raw"] / (CHONDRITE["La"] / CHONDRITE["Yb"])
    if {"Ce", "Y"}.issubset(out.columns): out["Ce_Y"] = safe_divide(out["Ce"], out["Y"])
    if {"Sm", "Yb"}.issubset(out.columns): out["Sm_Yb"] = safe_divide(out["Sm"], out["Yb"])
    if {"Gd", "Yb"}.issubset(out.columns): out["Gd_Yb"] = safe_divide(out["Gd"], out["Yb"])
    if {"Dy", "Yb"}.issubset(out.columns): out["Dy_Yb"] = safe_divide(out["Dy"], out["Yb"])
    if {"Rb", "Sr"}.issubset(out.columns): out["Rb_Sr"] = safe_divide(out["Rb"], out["Sr"])
    if {"MnO", "MgO"}.issubset(out.columns): out["MnO_MgO"] = safe_divide(out["MnO"], out["MgO"])
    if {"FeO", "MgO"}.issubset(out.columns): out["FeO_MgO"] = safe_divide(out["FeO"], out["MgO"])
    if {"Sm", "Eu", "Gd"}.issubset(out.columns):
        sm_n = pd.to_numeric(out["Sm"], errors="coerce") / CHONDRITE["Sm"]
        eu_n = pd.to_numeric(out["Eu"], errors="coerce") / CHONDRITE["Eu"]
        gd_n = pd.to_numeric(out["Gd"], errors="coerce") / CHONDRITE["Gd"]
        out["Eu_Eu_star"] = eu_n / np.sqrt(sm_n * gd_n)
    return out
